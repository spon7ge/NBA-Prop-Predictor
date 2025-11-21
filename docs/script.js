const prizepicksSinglesData = [
    {"name": "Tre Jones", "bookmaker": "FanDuel", "line": 8.5, "prediction": 14.89, "side": "Over", "odds": -106, "recommendation": 1, "ev": 6.79, "kelly": 0.72, "sigma": "Med"},
    {"name": "Dillon Brooks", "bookmaker": "BetMGM", "line": 16.5, "prediction": 22.8, "side": "Over", "odds": -115, "recommendation": 1, "ev": 5.52, "kelly": 0.635, "sigma": "High"},
    {"name": "Alperen Sengun", "bookmaker": "FanDuel", "line": 23.5, "prediction": 28.14, "side": "Over", "odds": -102, "recommendation": 1, "ev": 5.09, "kelly": 0.519, "sigma": "High"},
    {"name": "Bennedict Mathurin", "bookmaker": "DraftKings", "line": 21.5, "prediction": 26.11, "side": "Over", "odds": -103, "recommendation": 1, "ev": 4.99, "kelly": 0.514, "sigma": "High"},
    {"name": "Kevin Huerter", "bookmaker": "BetMGM", "line": 12.5, "prediction": 16.73, "side": "Over", "odds": -105, "recommendation": 1, "ev": 4.68, "kelly": 0.491, "sigma": "High"},
    {"name": "Isaac Okoro", "bookmaker": "DraftKings", "line": 7.5, "prediction": 11.22, "side": "Over", "odds": -102, "recommendation": 0, "ev": 4.52, "kelly": 0.461, "sigma": "Med"},
    {"name": "Darius Garland", "bookmaker": "BetRivers", "line": 15.5, "prediction": 12.47, "side": "Under", "odds": -107, "recommendation": 0, "ev": 4.25, "kelly": 0.455, "sigma": "Low"},
    {"name": "Keyonte George", "bookmaker": "FanDuel", "line": 19.5, "prediction": 23.11, "side": "Over", "odds": 100, "recommendation": 0, "ev": 4.03, "kelly": 0.403, "sigma": "High"},
    {"name": "Lauri Markkanen", "bookmaker": "BetRivers", "line": 24.5, "prediction": 28.53, "side": "Over", "odds": -107, "recommendation": 1, "ev": 3.93, "kelly": 0.42, "sigma": "High"},
    {"name": "Coby White", "bookmaker": "BetMGM", "line": 20.5, "prediction": 24.37, "side": "Over", "odds": -118, "recommendation": 0, "ev": 3.88, "kelly": 0.457, "sigma": "Med"},
];const prizepicksPairsData = [
    {"name1": "Tre Jones", "name2": "Dillon Brooks", "line1": 8.5, "line2": 16.5, "prediction1": 14.89, "prediction2": 22.8, "side1": "over", "side2": "over", "recommendation": 1, "ev": 11.08, "kelly": 0.554, "sigma1": "Med", "sigma2": "High", "prob1": 0.864, "prob2": 0.83, "hitRate1": 92.1, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 81.3, "l5_2": 0.8, "l15_2": 0.4},
    {"name1": "Kevin Huerter", "name2": "Alperen Sengun", "line1": 12.5, "line2": 22.5, "prediction1": 16.73, "prediction2": 28.14, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.84, "kelly": 0.392, "sigma1": "High", "sigma2": "High", "prob1": 0.752, "prob2": 0.807, "hitRate1": 66.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 58.0, "l5_2": 0.8, "l15_2": 0.47},
    {"name1": "Coby White", "name2": "Aaron Gordon", "line1": 20.5, "line2": 16.5, "prediction1": 24.37, "prediction2": 19.75, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.17, "kelly": 0.309, "sigma1": "Med", "sigma2": "Med", "prob1": 0.751, "prob2": 0.732, "hitRate1": 72.9, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 79.4, "l5_2": 1.0, "l15_2": 0.6},
    {"name1": "Khris Middleton", "name2": "Isaiah Collier", "line1": 9.5, "line2": 8.0, "prediction1": 12.73, "prediction2": 10.65, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.51, "kelly": 0.275, "sigma1": "Med", "sigma2": "Low", "prob1": 0.728, "prob2": 0.724, "hitRate1": 32.8, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 73.6, "l5_2": 0.8, "l15_2": 0.33},
    {"name1": "Julius Randle", "name2": "Lauri Markkanen", "line1": 22.5, "line2": 24.5, "prediction1": 26.32, "prediction2": 28.53, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.12, "kelly": 0.256, "sigma1": "High", "sigma2": "High", "prob1": 0.715, "prob2": 0.72, "hitRate1": 59.3, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 85.0, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Naji Marshall", "name2": "Keyonte George", "line1": 10.5, "line2": 19.5, "prediction1": 13.91, "prediction2": 23.11, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.71, "kelly": 0.236, "sigma1": "High", "sigma2": "High", "prob1": 0.713, "prob2": 0.702, "hitRate1": 90.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 75.3, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Klay Thompson", "name2": "Draymond Green", "line1": 11.5, "line2": 8.5, "prediction1": 8.45, "prediction2": 11.38, "side1": "under", "side2": "over", "recommendation": 0, "ev": 4.14, "kelly": 0.207, "sigma1": "Med", "sigma2": "Med", "prob1": 0.699, "prob2": 0.688, "hitRate1": 60.3, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 42.4, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "D'Angelo Russell", "name2": "Deni Avdija", "line1": 12.5, "line2": 25.5, "prediction1": 15.66, "prediction2": 29.02, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.76, "kelly": 0.188, "sigma1": "High", "sigma2": "High", "prob1": 0.684, "prob2": 0.684, "hitRate1": 44.1, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 63.8, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Bilal Coulibaly", "name2": "Jerami Grant", "line1": 10.5, "line2": 19.5, "prediction1": 13.33, "prediction2": 17.1, "side1": "over", "side2": "under", "recommendation": 0, "ev": 3.54, "kelly": 0.177, "sigma1": "Med", "sigma2": "Med", "prob1": 0.682, "prob2": 0.675, "hitRate1": 42.9, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 55.2, "l5_2": 0.4, "l15_2": 0.4},
    {"name1": "Darius Garland", "name2": "Brandon Williams", "line1": 14.5, "line2": 13.5, "prediction1": 12.47, "prediction2": 16.4, "side1": "under", "side2": "over", "recommendation": 0, "ev": 3.14, "kelly": 0.157, "sigma1": "Low", "sigma2": "High", "prob1": 0.664, "prob2": 0.673, "hitRate1": 20.3, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 41.9, "l5_2": 0.6, "l15_2": 0.4},
];const prizepicksTriosData = [
    {"name1": "Tre Jones", "name2": "Dillon Brooks", "name3": "Alperen Sengun", "line1": 8.5, "line2": 16.5, "line3": 22.5, "prediction1": 14.89, "prediction2": 22.8, "prediction3": 28.14, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 21.25, "kelly": 0.425, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.864, "prob2": 0.83, "prob3": 0.807, "hitRate1": 92.1, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 81.3, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 58.0, "l5_3": 0.8, "l15_3": 0.47},
    {"name1": "Coby White", "name2": "Kevin Huerter", "name3": "Aaron Gordon", "line1": 20.5, "line2": 12.5, "line3": 16.5, "prediction1": 24.37, "prediction2": 16.73, "prediction3": 19.75, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 12.33, "kelly": 0.247, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.751, "prob2": 0.752, "prob3": 0.732, "hitRate1": 72.9, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 66.4, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 79.4, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Khris Middleton", "name2": "Isaiah Collier", "name3": "Lauri Markkanen", "line1": 9.5, "line2": 8.0, "line3": 24.5, "prediction1": 12.73, "prediction2": 10.65, "prediction3": 28.53, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 10.5, "kelly": 0.21, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.728, "prob2": 0.724, "prob3": 0.72, "hitRate1": 32.8, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 73.6, "l5_2": 0.8, "l15_2": 0.33, "hitRate3": 85.0, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Naji Marshall", "name2": "Julius Randle", "name3": "Keyonte George", "line1": 10.5, "line2": 22.5, "line3": 19.5, "prediction1": 13.91, "prediction2": 26.32, "prediction3": 23.11, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.31, "kelly": 0.186, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.713, "prob2": 0.715, "prob3": 0.702, "hitRate1": 90.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 59.3, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 75.3, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "D'Angelo Russell", "name2": "Klay Thompson", "name3": "Draymond Green", "line1": 12.5, "line2": 11.5, "line3": 8.5, "prediction1": 15.66, "prediction2": 8.45, "prediction3": 11.38, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 7.77, "kelly": 0.155, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.684, "prob2": 0.699, "prob3": 0.688, "hitRate1": 44.1, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 60.3, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 42.4, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Bilal Coulibaly", "name2": "Deni Avdija", "name3": "Jerami Grant", "line1": 10.5, "line2": 25.5, "line3": 19.5, "prediction1": 13.33, "prediction2": 29.02, "prediction3": 17.1, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 7.01, "kelly": 0.14, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.682, "prob2": 0.684, "prob3": 0.675, "hitRate1": 42.9, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 63.8, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 55.2, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Darius Garland", "name2": "Josh Giddey", "name3": "Brandon Williams", "line1": 14.5, "line2": 20.5, "line3": 13.5, "prediction1": 12.47, "prediction2": 23.22, "prediction3": 16.4, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.93, "kelly": 0.119, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.664, "prob2": 0.66, "prob3": 0.673, "hitRate1": 20.3, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 51.1, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 41.9, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Jeremiah Fears", "name2": "Cameron Johnson", "name3": "Shai Gilgeous-Alexander", "line1": 14.5, "line2": 11.5, "line3": 30.5, "prediction1": 17.24, "prediction2": 9.26, "prediction3": 28.1, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 5.4, "kelly": 0.108, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.656, "prob2": 0.66, "prob3": 0.658, "hitRate1": 70.0, "l5_1": 1.0, "l15_1": 0.67, "hitRate2": 82.6, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 49.9, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Matas Buzelis", "name2": "Anthony Edwards", "name3": "Nikola Joki\u0107", "line1": 14.5, "line2": 28.5, "line3": 28.5, "prediction1": 17.2, "prediction2": 26.21, "prediction3": 30.83, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 5.04, "kelly": 0.101, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.652, "prob2": 0.654, "prob3": 0.653, "hitRate1": 55.9, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 63.6, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Andrew Nembhard", "name2": "Isaiah Jackson", "name3": "Saddiq Bey", "line1": 16.5, "line2": 7.5, "line3": 9.5, "prediction1": 19.03, "prediction2": 9.57, "prediction3": 11.87, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 4.38, "kelly": 0.088, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.641, "prob2": 0.645, "prob3": 0.644, "hitRate1": 68.5, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 60.1, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 86.7, "l5_3": 0.8, "l15_3": 0.6},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Tre Jones", "name2": "Dillon Brooks", "line1": 8.5, "line2": 17.5, "prediction1": 14.89, "prediction2": 22.8, "side1": "over", "side2": "over", "recommendation": 1, "ev": 10.04, "kelly": 0.502, "sigma1": "Med", "sigma2": "High", "prob1": 0.864, "prob2": 0.789, "hitRate1": 92.1, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 74.4, "l5_2": 0.8, "l15_2": 0.4},
    {"name1": "Isaac Okoro", "name2": "Alperen Sengun", "line1": 6.5, "line2": 23.5, "prediction1": 11.22, "prediction2": 28.14, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.59, "kelly": 0.38, "sigma1": "Med", "sigma2": "High", "prob1": 0.785, "prob2": 0.762, "hitRate1": 76.1, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 49.8, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Bennedict Mathurin", "name2": "Kevin Huerter", "line1": 21.5, "line2": 12.5, "prediction1": 26.11, "prediction2": 16.73, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.82, "kelly": 0.341, "sigma1": "High", "sigma2": "High", "prob1": 0.761, "prob2": 0.752, "hitRate1": 41.3, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 66.4, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Darius Garland", "name2": "Coby White", "line1": 15.5, "line2": 20.5, "prediction1": 12.47, "prediction2": 24.37, "side1": "under", "side2": "over", "recommendation": 0, "ev": 6.27, "kelly": 0.313, "sigma1": "Low", "sigma2": "Med", "prob1": 0.737, "prob2": 0.751, "hitRate1": 28.1, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 72.9, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Julius Randle", "name2": "Lauri Markkanen", "line1": 22.5, "line2": 24.5, "prediction1": 26.32, "prediction2": 28.53, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.12, "kelly": 0.256, "sigma1": "High", "sigma2": "High", "prob1": 0.715, "prob2": 0.72, "hitRate1": 59.3, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 85.0, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Naji Marshall", "name2": "Keyonte George", "line1": 10.5, "line2": 19.5, "prediction1": 13.91, "prediction2": 23.11, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.71, "kelly": 0.236, "sigma1": "High", "sigma2": "High", "prob1": 0.713, "prob2": 0.702, "hitRate1": 90.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 75.3, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Klay Thompson", "name2": "Draymond Green", "line1": 11.5, "line2": 8.5, "prediction1": 8.45, "prediction2": 11.38, "side1": "under", "side2": "over", "recommendation": 0, "ev": 4.14, "kelly": 0.207, "sigma1": "Med", "sigma2": "Med", "prob1": 0.699, "prob2": 0.688, "hitRate1": 60.3, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 42.4, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Bilal Coulibaly", "name2": "Deni Avdija", "line1": 10.5, "line2": 25.5, "prediction1": 13.33, "prediction2": 29.02, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.72, "kelly": 0.186, "sigma1": "Med", "sigma2": "High", "prob1": 0.682, "prob2": 0.684, "hitRate1": 42.9, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 63.8, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Josh Giddey", "name2": "Aaron Gordon", "line1": 20.5, "line2": 17.5, "prediction1": 23.22, "prediction2": 19.75, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.93, "kelly": 0.146, "sigma1": "High", "sigma2": "Med", "prob1": 0.66, "prob2": 0.666, "hitRate1": 51.1, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 72.1, "l5_2": 0.8, "l15_2": 0.47},
    {"name1": "Cameron Johnson", "name2": "Shai Gilgeous-Alexander", "line1": 11.5, "line2": 30.5, "prediction1": 9.26, "prediction2": 28.1, "side1": "under", "side2": "under", "recommendation": 0, "ev": 2.77, "kelly": 0.139, "sigma1": "Med", "sigma2": "Med", "prob1": 0.66, "prob2": 0.658, "hitRate1": 82.6, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 49.9, "l5_2": 0.4, "l15_2": 0.47},
];const underdogTriosData = [
    {"name1": "Tre Jones", "name2": "Isaac Okoro", "name3": "Dillon Brooks", "line1": 8.5, "line2": 6.5, "line3": 17.5, "prediction1": 14.89, "prediction2": 11.22, "prediction3": 22.8, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 18.91, "kelly": 0.378, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.864, "prob2": 0.785, "prob3": 0.789, "hitRate1": 92.1, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 76.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 74.4, "l5_3": 0.8, "l15_3": 0.4},
    {"name1": "Bennedict Mathurin", "name2": "Kevin Huerter", "name3": "Alperen Sengun", "line1": 21.5, "line2": 12.5, "line3": 23.5, "prediction1": 26.11, "prediction2": 16.73, "prediction3": 28.14, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 13.54, "kelly": 0.271, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.761, "prob2": 0.752, "prob3": 0.762, "hitRate1": 41.3, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 66.4, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 49.8, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Darius Garland", "name2": "Coby White", "name3": "Lauri Markkanen", "line1": 15.5, "line2": 20.5, "line3": 24.5, "prediction1": 12.47, "prediction2": 24.37, "prediction3": 28.53, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.51, "kelly": 0.23, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "prob1": 0.737, "prob2": 0.751, "prob3": 0.72, "hitRate1": 28.1, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 72.9, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 85.0, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Naji Marshall", "name2": "Julius Randle", "name3": "Keyonte George", "line1": 10.5, "line2": 22.5, "line3": 19.5, "prediction1": 13.91, "prediction2": 26.32, "prediction3": 23.11, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.31, "kelly": 0.186, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.713, "prob2": 0.715, "prob3": 0.702, "hitRate1": 90.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 59.3, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 75.3, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Bilal Coulibaly", "name2": "Klay Thompson", "name3": "Draymond Green", "line1": 10.5, "line2": 11.5, "line3": 8.5, "prediction1": 13.33, "prediction2": 8.45, "prediction3": 11.38, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 7.72, "kelly": 0.154, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "prob1": 0.682, "prob2": 0.699, "prob3": 0.688, "hitRate1": 42.9, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 60.3, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 42.4, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Josh Giddey", "name2": "Aaron Gordon", "name3": "Deni Avdija", "line1": 20.5, "line2": 17.5, "line3": 25.5, "prediction1": 23.22, "prediction2": 19.75, "prediction3": 29.02, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.25, "kelly": 0.125, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.66, "prob2": 0.666, "prob3": 0.684, "hitRate1": 51.1, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 72.1, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 63.8, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Jeremiah Fears", "name2": "Cameron Johnson", "name3": "Shai Gilgeous-Alexander", "line1": 14.5, "line2": 11.5, "line3": 30.5, "prediction1": 17.24, "prediction2": 9.26, "prediction3": 28.1, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 5.4, "kelly": 0.108, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.656, "prob2": 0.66, "prob3": 0.658, "hitRate1": 70.0, "l5_1": 1.0, "l15_1": 0.67, "hitRate2": 82.6, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 49.9, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Isaiah Jackson", "name2": "Saddiq Bey", "name3": "Anthony Edwards", "line1": 7.5, "line2": 9.5, "line3": 28.5, "prediction1": 9.57, "prediction2": 11.87, "prediction3": 26.21, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 4.67, "kelly": 0.093, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.645, "prob2": 0.644, "prob3": 0.654, "hitRate1": 60.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 86.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 63.6, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Donovan Mitchell", "name2": "Jaylon Tyson", "name3": "Brandon Ingram", "line1": 28.5, "line2": 8.5, "line3": 21.5, "prediction1": 26.01, "prediction2": 10.75, "prediction3": 23.71, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 4.12, "kelly": 0.082, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.641, "prob2": 0.642, "prob3": 0.635, "hitRate1": 52.2, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 93.0, "l5_2": 1.0, "l15_2": 0.47, "hitRate3": 41.9, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Andrew Nembhard", "name2": "Simone Fontecchio", "name3": "Yves Missi", "line1": 16.5, "line2": 10.5, "line3": 7.5, "prediction1": 19.03, "prediction2": 12.38, "prediction3": 5.93, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 3.62, "kelly": 0.072, "sigma1": "High", "sigma2": "Med", "sigma3": "Low", "prob1": 0.641, "prob2": 0.63, "prob3": 0.625, "hitRate1": 68.5, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 51.3, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 88.0, "l5_3": 0.2, "l15_3": 0.2},
];const prizepicksPointsHitRates = [
    {"name": "Jaylon Tyson", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.93, "underPct": 0.07},
    {"name": "Tre Jones", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.921, "underPct": 0.079},
    {"name": "Naji Marshall", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.901, "underPct": 0.099},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.883, "underPct": 0.117},
    {"name": "Trey Murphy III", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.872, "underPct": 0.128},
    {"name": "Saddiq Bey", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.867, "underPct": 0.133},
    {"name": "Lauri Markkanen", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.85, "underPct": 0.15},
    {"name": "Dillon Brooks", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.813, "underPct": 0.187},
    {"name": "Sandro Mamukelashvili", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.809, "underPct": 0.191},
    {"name": "Darius Garland", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.797, "underPct": 0.203},
    {"name": "Aaron Gordon", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.794, "underPct": 0.206},
    {"name": "Ayo Dosunmu", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.794, "underPct": 0.206},
    {"name": "Keyonte George", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.753, "underPct": 0.247},
    {"name": "Isaiah Collier", "line": 8.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.736, "underPct": 0.264},
    {"name": "Coby White", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.729, "underPct": 0.271},
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.718, "underPct": 0.282},
    {"name": "Jalen Smith", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.705, "underPct": 0.295},
    {"name": "Reed Sheppard", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.703, "underPct": 0.297},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.7, "underPct": 0.3},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.685, "underPct": 0.315},
    {"name": "Josh Minott", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.681, "underPct": 0.319},
    {"name": "Shaedon Sharpe", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.667, "underPct": 0.333},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.664, "underPct": 0.336},
    {"name": "Jaden McDaniels", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.657, "underPct": 0.343},
    {"name": "Noah Clowney", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.654, "underPct": 0.346},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.638, "underPct": 0.362},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.634, "underPct": 0.366},
    {"name": "Jakob Poeltl", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.632, "underPct": 0.368},
    {"name": "Stephen Curry", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.625, "underPct": 0.375},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.615, "underPct": 0.385},
    {"name": "Day'Ron Sharpe", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.603, "underPct": 0.397},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.603, "underPct": 0.397},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.601, "underPct": 0.399},
    {"name": "Tre Johnson", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.598, "underPct": 0.402},
    {"name": "Pelle Larsson", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.594, "underPct": 0.406},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.593, "underPct": 0.407},
    {"name": "Payton Pritchard", "line": 17.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.593, "underPct": 0.407},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
    {"name": "Ace Bailey", "line": 11.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.584, "underPct": 0.416},
    {"name": "De'Andre Hunter", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.584, "underPct": 0.416},
    {"name": "Alperen Sengun", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.58, "underPct": 0.42},
    {"name": "Luka Garza", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.579, "underPct": 0.421},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.578, "underPct": 0.422},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.559, "underPct": 0.441},
    {"name": "Corey Kispert", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.557, "underPct": 0.443},
    {"name": "Cam Whitmore", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.555, "underPct": 0.445},
    {"name": "Jaylen Brown", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.55, "underPct": 0.45},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.548, "underPct": 0.452},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.547, "underPct": 0.453},
    {"name": "Jeremiah Robinson-Earl", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "Jamal Murray", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.531, "underPct": 0.469},
    {"name": "Dru Smith", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.528, "underPct": 0.472},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Tyrese Martin", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.52, "underPct": 0.48},
    {"name": "Simone Fontecchio", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.513, "underPct": 0.487},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.511, "underPct": 0.489},
    {"name": "Alex Sarr", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Jarace Walker", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.499, "underPct": 0.501},
    {"name": "Brandin Podziemski", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.499, "underPct": 0.501},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.495, "underPct": 0.505},
    {"name": "Isaiah Joe", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.492, "underPct": 0.508},
    {"name": "Derik Queen", "line": 14.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.487, "underPct": 0.513},
    {"name": "Zion Williamson", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.486, "underPct": 0.514},
    {"name": "Jrue Holiday", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.483, "underPct": 0.517},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.478, "underPct": 0.522},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.472, "underPct": 0.528},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.468, "underPct": 0.532},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.464, "underPct": 0.536},
    {"name": "Will Richard", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.461, "underPct": 0.539},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.46, "underPct": 0.54},
    {"name": "Anfernee Simons", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.454, "underPct": 0.546},
    {"name": "Bam Adebayo", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.451, "underPct": 0.549},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Jerami Grant", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.448, "underPct": 0.552},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.441, "underPct": 0.559},
    {"name": "Ajay Mitchell", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.433, "underPct": 0.567},
    {"name": "Mark Williams", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.429, "underPct": 0.571},
    {"name": "Luguentz Dort", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.427, "underPct": 0.573},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.425, "underPct": 0.575},
    {"name": "Pascal Siakam", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.425, "underPct": 0.575},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.424, "underPct": 0.576},
    {"name": "Brandon Ingram", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.419, "underPct": 0.581},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.419, "underPct": 0.581},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.419, "underPct": 0.581},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.416, "underPct": 0.584},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.414, "underPct": 0.586},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.412, "underPct": 0.588},
    {"name": "Moses Moody", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.406, "underPct": 0.594},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.401, "underPct": 0.599},
    {"name": "Klay Thompson", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.397, "underPct": 0.603},
    {"name": "Jarrett Allen", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.395, "underPct": 0.605},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.394, "underPct": 0.606},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.38, "underPct": 0.62},
    {"name": "Oso Ighodaro", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.373, "underPct": 0.627},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.371, "underPct": 0.629},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.364, "underPct": 0.636},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.348, "underPct": 0.652},
    {"name": "Khris Middleton", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.328, "underPct": 0.672},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.285, "underPct": 0.715},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.265, "underPct": 0.735},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.239, "underPct": 0.761},
    {"name": "Al Horford", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.236, "underPct": 0.764},
    {"name": "Terance Mann", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.218, "underPct": 0.782},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.215, "underPct": 0.785},
    {"name": "Ben Sheppard", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.214, "underPct": 0.786},
    {"name": "Ryan Dunn", "line": 9.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.201, "underPct": 0.799},
    {"name": "Collin Gillespie", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.183, "underPct": 0.817},
    {"name": "Cameron Johnson", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.174, "underPct": 0.826},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.162, "underPct": 0.838},
    {"name": "Dereck Lively II", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.15, "underPct": 0.85},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.127, "underPct": 0.873},
];const prizepicksAssistsHitRates = [
    {"name": "Coby White", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.772, "underPct": 0.228},
    {"name": "Darius Garland", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.752, "underPct": 0.248},
    {"name": "Josh Giddey", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.674, "underPct": 0.326},
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.645, "underPct": 0.355},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.62, "underPct": 0.38},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.617, "underPct": 0.383},
    {"name": "Jrue Holiday", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.608, "underPct": 0.392},
    {"name": "Isaiah Collier", "line": 6.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.596, "underPct": 0.404},
    {"name": "Derrick White", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.569, "underPct": 0.431},
    {"name": "Derik Queen", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Jarrett Allen", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.554, "underPct": 0.446},
    {"name": "Kyshawn George", "line": 4.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.536, "underPct": 0.464},
    {"name": "Deni Avdija", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.525, "underPct": 0.475},
    {"name": "Buddy Hield", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.517, "underPct": 0.483},
    {"name": "Kel'el Ware", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.513, "underPct": 0.487},
    {"name": "Keyonte George", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.511, "underPct": 0.489},
    {"name": "Norman Powell", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.498, "underPct": 0.502},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.497, "underPct": 0.503},
    {"name": "Jaden McDaniels", "line": 2.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.497, "underPct": 0.503},
    {"name": "Zion Williamson", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.489, "underPct": 0.511},
    {"name": "Brandon Williams", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.485, "underPct": 0.515},
    {"name": "Alperen Sengun", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.479, "underPct": 0.521},
    {"name": "Jamal Murray", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.477, "underPct": 0.523},
    {"name": "Andrew Nembhard", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.469, "underPct": 0.531},
    {"name": "Donovan Mitchell", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.46, "underPct": 0.54},
    {"name": "Pascal Siakam", "line": 4.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.46, "underPct": 0.54},
    {"name": "T.J. McConnell", "line": 3.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.446, "underPct": 0.554},
    {"name": "Max Christie", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.437, "underPct": 0.563},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Scottie Barnes", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.429, "underPct": 0.571},
    {"name": "Draymond Green", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.385, "underPct": 0.615},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.384, "underPct": 0.616},
    {"name": "Amen Thompson", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.381, "underPct": 0.619},
    {"name": "Collin Gillespie", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.374, "underPct": 0.626},
    {"name": "Ajay Mitchell", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.371, "underPct": 0.629},
    {"name": "Devin Booker", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.368, "underPct": 0.632},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.368, "underPct": 0.632},
    {"name": "Davion Mitchell", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.346, "underPct": 0.654},
    {"name": "Jeremiah Fears", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.345, "underPct": 0.655},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.342, "underPct": 0.658},
    {"name": "Anthony Edwards", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.34, "underPct": 0.66},
    {"name": "Naji Marshall", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.33, "underPct": 0.67},
    {"name": "Shai Gilgeous-Alexander", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.33, "underPct": 0.67},
    {"name": "Jamal Shead", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.32, "underPct": 0.68},
    {"name": "Payton Pritchard", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.32, "underPct": 0.68},
    {"name": "Pelle Larsson", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.276, "underPct": 0.724},
    {"name": "Stephen Curry", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.263, "underPct": 0.737},
    {"name": "D'Angelo Russell", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.24, "underPct": 0.76},
];const prizepicksReboundsHitRates = [
    {"name": "Kel'el Ware", "line": 9.0, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.807, "underPct": 0.193},
    {"name": "Tre Jones", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.801, "underPct": 0.199},
    {"name": "Donovan Mitchell", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.739, "underPct": 0.261},
    {"name": "Naji Marshall", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.689, "underPct": 0.311},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.682, "underPct": 0.318},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.669, "underPct": 0.331},
    {"name": "Matas Buzelis", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.628, "underPct": 0.372},
    {"name": "Alperen Sengun", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
    {"name": "Isaiah Joe", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.584, "underPct": 0.416},
    {"name": "Jalen Smith", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.568, "underPct": 0.432},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.556, "underPct": 0.444},
    {"name": "Luguentz Dort", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.549, "underPct": 0.451},
    {"name": "Jaylon Tyson", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.538, "underPct": 0.462},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.524, "underPct": 0.476},
    {"name": "Amen Thompson", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.521, "underPct": 0.479},
    {"name": "Scottie Barnes", "line": 8.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.52, "underPct": 0.48},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.52, "underPct": 0.48},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.517, "underPct": 0.483},
    {"name": "Naz Reid", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.511, "underPct": 0.489},
    {"name": "Zion Williamson", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.507, "underPct": 0.493},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.506, "underPct": 0.494},
    {"name": "Isaiah Hartenstein", "line": 10.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.492, "underPct": 0.508},
    {"name": "Shai Gilgeous-Alexander", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.486, "underPct": 0.514},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.484, "underPct": 0.516},
    {"name": "Jaylen Brown", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.478, "underPct": 0.522},
    {"name": "Immanuel Quickley", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.475, "underPct": 0.525},
    {"name": "Derik Queen", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.473, "underPct": 0.527},
    {"name": "Collin Gillespie", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.473, "underPct": 0.527},
    {"name": "Isaiah Jackson", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.472, "underPct": 0.528},
    {"name": "Aaron Gordon", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.472, "underPct": 0.528},
    {"name": "Neemias Queta", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.459, "underPct": 0.541},
    {"name": "Devin Booker", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.455, "underPct": 0.545},
    {"name": "Derrick White", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.448, "underPct": 0.552},
    {"name": "Chet Holmgren", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.438, "underPct": 0.562},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.436, "underPct": 0.564},
    {"name": "Cooper Flagg", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.434, "underPct": 0.566},
    {"name": "Alex Sarr", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.431, "underPct": 0.569},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.425, "underPct": 0.575},
    {"name": "Shaedon Sharpe", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.419, "underPct": 0.581},
    {"name": "Jrue Holiday", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.416, "underPct": 0.584},
    {"name": "Jarace Walker", "line": 4.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.41, "underPct": 0.59},
    {"name": "Evan Mobley", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.408, "underPct": 0.592},
    {"name": "De'Andre Hunter", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.407, "underPct": 0.593},
    {"name": "Jakob Poeltl", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.406, "underPct": 0.594},
    {"name": "Royce O'Neale", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Norman Powell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.398, "underPct": 0.602},
    {"name": "P.J. Washington", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.396, "underPct": 0.604},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.396, "underPct": 0.604},
    {"name": "Deni Avdija", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.383, "underPct": 0.617},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.379, "underPct": 0.621},
    {"name": "Yves Missi", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.373, "underPct": 0.627},
    {"name": "Ace Bailey", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.371, "underPct": 0.629},
    {"name": "Bennedict Mathurin", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.37, "underPct": 0.63},
    {"name": "Kevin Durant", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.369, "underPct": 0.631},
    {"name": "Luka Garza", "line": 5.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.368, "underPct": 0.632},
    {"name": "Josh Minott", "line": 4.0, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.348, "underPct": 0.652},
    {"name": "Pascal Siakam", "line": 6.0, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.345, "underPct": 0.655},
    {"name": "Will Richard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.342, "underPct": 0.658},
    {"name": "Klay Thompson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.335, "underPct": 0.665},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.327, "underPct": 0.673},
    {"name": "Daniel Gafford", "line": 7.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.327, "underPct": 0.673},
    {"name": "Dereck Lively II", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.326, "underPct": 0.674},
    {"name": "T.J. McConnell", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.323, "underPct": 0.677},
    {"name": "Ben Sheppard", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.4, "overPct": 0.294, "underPct": 0.706},
    {"name": "Draymond Green", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.292, "underPct": 0.708},
    {"name": "Lauri Markkanen", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.287, "underPct": 0.713},
    {"name": "Al Horford", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.278, "underPct": 0.722},
    {"name": "Bilal Coulibaly", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.276, "underPct": 0.724},
    {"name": "Sandro Mamukelashvili", "line": 5.0, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.26, "underPct": 0.74},
    {"name": "Jarrett Allen", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.234, "underPct": 0.766},
    {"name": "Mark Williams", "line": 9.0, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.234, "underPct": 0.766},
    {"name": "Stephen Curry", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.223, "underPct": 0.777},
    {"name": "Khris Middleton", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.22, "underPct": 0.78},
    {"name": "Noah Clowney", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.217, "underPct": 0.783},
    {"name": "Kyshawn George", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.212, "underPct": 0.788},
    {"name": "Ryan Dunn", "line": 5.0, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.209, "underPct": 0.791},
    {"name": "Pelle Larsson", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.201, "underPct": 0.799},
];const prizepicksBlocksHitRates = [
    {"name": "Scottie Barnes", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.614, "underPct": 0.386},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.537, "underPct": 0.463},
    {"name": "Cooper Flagg", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.558, "underPct": 0.442},
    {"name": "Zion Williamson", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.278, "underPct": 0.722},
    {"name": "Amen Thompson", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.397, "underPct": 0.603},
    {"name": "Kevin Durant", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.505, "underPct": 0.495},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.49, "underPct": 0.51},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.507, "underPct": 0.493},
    {"name": "Isaiah Hartenstein", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
];const prizepicksStealsHitRates = [
    {"name": "Jarace Walker", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.411, "underPct": 0.589},
    {"name": "T.J. McConnell", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.444, "underPct": 0.556},
    {"name": "Day'Ron Sharpe", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.596, "underPct": 0.404},
    {"name": "Neemias Queta", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.451, "underPct": 0.549},
    {"name": "Drake Powell", "line": 0.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.579, "underPct": 0.421},
    {"name": "Josh Minott", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.512, "underPct": 0.488},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.393, "underPct": 0.607},
    {"name": "D'Angelo Russell", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.391, "underPct": 0.609},
    {"name": "Dereck Lively II", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.402, "underPct": 0.598},
    {"name": "Saddiq Bey", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.459, "underPct": 0.541},
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.531, "underPct": 0.469},
    {"name": "Al Horford", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.623, "underPct": 0.377},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.572, "underPct": 0.428},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Jaylon Tyson", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kel'el Ware", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 36.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Day'Ron Sharpe", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 40.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Simone Fontecchio", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dillon Brooks", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 38.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 29.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jrue Holiday", "line": 25.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tre Jones", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shaedon Sharpe", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaac Okoro", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 34.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cameron Johnson", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 36.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Gordon", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Oso Ighodaro", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Al Horford", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rudy Gobert", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Huerter", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Whitmore", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 27.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 28.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Jackson", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandin Podziemski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 39.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Moses Moody", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 24.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Minott", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dru Smith", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Khris Middleton", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Dunn", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Alex Sarr", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 38.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dereck Lively II", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pelle Larsson", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarrett Allen", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Darius Garland", "line": 22.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ben Sheppard", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 39.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "P.J. Washington", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 28.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPRHitRates = [
    {"name": "Kel'el Ware", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tre Jones", "line": 11.5, "l5": 1.0, "l10": 1.0, "l15": 0.73, "overPct": 1.0, "underPct": 0.0},
    {"name": "Jaylon Tyson", "line": 13.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ziaire Williams", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 14.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luka Garza", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shaedon Sharpe", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jrue Holiday", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaac Okoro", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Simone Fontecchio", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Daniel Gafford", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 15.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Svi Mykhailiuk", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tre Johnson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Neemias Queta", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Robinson-Earl", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sandro Mamukelashvili", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Whitmore", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Jackson", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Moses Moody", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ben Sheppard", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Al Horford", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dereck Lively II", "line": 13.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Khris Middleton", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pelle Larsson", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 11.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alex Sarr", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Minott", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Day'Ron Sharpe", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Collin Gillespie", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Darius Garland", "line": 17.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jamal Shead", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "P.J. Washington", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jarrett Allen", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 21.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 33.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Dunn", "line": 14.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPAHitRates = [
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Jones", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaden McDaniels", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Luka Garza", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Simone Fontecchio", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 21.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jrue Holiday", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "T.J. McConnell", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Norman Powell", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Dunn", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Buddy Hield", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Khris Middleton", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Coby White", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bam Adebayo", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaac Okoro", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pelle Larsson", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dean Wade", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Daniel Gafford", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ben Sheppard", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gradey Dick", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Darius Garland", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Dereck Lively II", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 35.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 19.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const prizepicksRAHitRates = [
    {"name": "Kel'el Ware", "line": 10.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 17.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 9.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 10.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 8.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zion Williamson", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Gordon", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 12.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jrue Holiday", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 14.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Garza", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 11.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 13.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 9.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Draymond Green", "line": 12.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Hartenstein", "line": 13.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Norman Powell", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pelle Larsson", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Coby White", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Simone Fontecchio", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ayo Dosunmu", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Nembhard", "line": 9.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 10.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Whitmore", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sandro Mamukelashvili", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Royce O'Neale", "line": 8.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Corey Kispert", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Khris Middleton", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 9.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const prizepicksTurnoversHitRates = [
    {"name": "Derrick White", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jrue Holiday", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sandro Mamukelashvili", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Corey Kispert", "line": 0.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luka Garza", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bilal Coulibaly", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Aaron Gordon", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Minott", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Josh Giddey", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jrue Holiday", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 2.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Derik Queen", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Robinson-Earl", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dru Smith", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 1.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogPointsHitRates = [
    {"name": "Jaylon Tyson", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.93, "underPct": 0.07},
    {"name": "Tre Jones", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.921, "underPct": 0.079},
    {"name": "Naji Marshall", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.901, "underPct": 0.099},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.883, "underPct": 0.117},
    {"name": "Trey Murphy III", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.872, "underPct": 0.128},
    {"name": "Saddiq Bey", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.867, "underPct": 0.133},
    {"name": "Lauri Markkanen", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.85, "underPct": 0.15},
    {"name": "Sandro Mamukelashvili", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.809, "underPct": 0.191},
    {"name": "Ayo Dosunmu", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.794, "underPct": 0.206},
    {"name": "Isaac Okoro", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.761, "underPct": 0.239},
    {"name": "Keyonte George", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.753, "underPct": 0.247},
    {"name": "Dillon Brooks", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.744, "underPct": 0.256},
    {"name": "Coby White", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.729, "underPct": 0.271},
    {"name": "Aaron Gordon", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.721, "underPct": 0.279},
    {"name": "Darius Garland", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.719, "underPct": 0.281},
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.718, "underPct": 0.282},
    {"name": "Jalen Smith", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.705, "underPct": 0.295},
    {"name": "Reed Sheppard", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.703, "underPct": 0.297},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.7, "underPct": 0.3},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.685, "underPct": 0.315},
    {"name": "Josh Minott", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.681, "underPct": 0.319},
    {"name": "Shaedon Sharpe", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.667, "underPct": 0.333},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.664, "underPct": 0.336},
    {"name": "Jaden McDaniels", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.657, "underPct": 0.343},
    {"name": "Noah Clowney", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.654, "underPct": 0.346},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.638, "underPct": 0.362},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.634, "underPct": 0.366},
    {"name": "Jakob Poeltl", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.632, "underPct": 0.368},
    {"name": "Stephen Curry", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.625, "underPct": 0.375},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.624, "underPct": 0.376},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.615, "underPct": 0.385},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.603, "underPct": 0.397},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.601, "underPct": 0.399},
    {"name": "Tre Johnson", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.598, "underPct": 0.402},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.593, "underPct": 0.407},
    {"name": "Payton Pritchard", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.593, "underPct": 0.407},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
    {"name": "De'Andre Hunter", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.584, "underPct": 0.416},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.584, "underPct": 0.416},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.578, "underPct": 0.422},
    {"name": "Corey Kispert", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.557, "underPct": 0.443},
    {"name": "Oso Ighodaro", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.548, "underPct": 0.452},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.548, "underPct": 0.452},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.547, "underPct": 0.453},
    {"name": "Jeremiah Robinson-Earl", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "Jamal Murray", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.531, "underPct": 0.469},
    {"name": "Jose Alvarado", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.527, "underPct": 0.473},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Simone Fontecchio", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.513, "underPct": 0.487},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.511, "underPct": 0.489},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.51, "underPct": 0.49},
    {"name": "Alex Sarr", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Jarace Walker", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.499, "underPct": 0.501},
    {"name": "Brandin Podziemski", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.499, "underPct": 0.501},
    {"name": "Alperen Sengun", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.498, "underPct": 0.502},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.495, "underPct": 0.505},
    {"name": "Derik Queen", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.487, "underPct": 0.513},
    {"name": "Zion Williamson", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.486, "underPct": 0.514},
    {"name": "Jrue Holiday", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.483, "underPct": 0.517},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.478, "underPct": 0.522},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.472, "underPct": 0.528},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.468, "underPct": 0.532},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.464, "underPct": 0.536},
    {"name": "Will Richard", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.461, "underPct": 0.539},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.46, "underPct": 0.54},
    {"name": "Bam Adebayo", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.451, "underPct": 0.549},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.429, "underPct": 0.571},
    {"name": "Luguentz Dort", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.427, "underPct": 0.573},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.425, "underPct": 0.575},
    {"name": "Pascal Siakam", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.425, "underPct": 0.575},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.424, "underPct": 0.576},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.419, "underPct": 0.581},
    {"name": "Brandon Ingram", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.419, "underPct": 0.581},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.416, "underPct": 0.584},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.413, "underPct": 0.587},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.412, "underPct": 0.588},
    {"name": "Moses Moody", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.406, "underPct": 0.594},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.401, "underPct": 0.599},
    {"name": "Klay Thompson", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.397, "underPct": 0.603},
    {"name": "Jarrett Allen", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.395, "underPct": 0.605},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.394, "underPct": 0.606},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.38, "underPct": 0.62},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.371, "underPct": 0.629},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.364, "underPct": 0.636},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.348, "underPct": 0.652},
    {"name": "Anfernee Simons", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.347, "underPct": 0.653},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.285, "underPct": 0.715},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.265, "underPct": 0.735},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.239, "underPct": 0.761},
    {"name": "Al Horford", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.236, "underPct": 0.764},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.215, "underPct": 0.785},
    {"name": "Ben Sheppard", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.214, "underPct": 0.786},
    {"name": "Ryan Dunn", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.201, "underPct": 0.799},
    {"name": "Collin Gillespie", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.183, "underPct": 0.817},
    {"name": "Cameron Johnson", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.174, "underPct": 0.826},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.162, "underPct": 0.838},
    {"name": "Dereck Lively II", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.15, "underPct": 0.85},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.127, "underPct": 0.873},
    {"name": "Yves Missi", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.12, "underPct": 0.88},
];const underdogAssistsHitRates = [
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.645, "underPct": 0.355},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.62, "underPct": 0.38},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.617, "underPct": 0.383},
    {"name": "Derik Queen", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Jarrett Allen", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.554, "underPct": 0.446},
    {"name": "Deni Avdija", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.525, "underPct": 0.475},
    {"name": "Buddy Hield", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.517, "underPct": 0.483},
    {"name": "Norman Powell", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.498, "underPct": 0.502},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.497, "underPct": 0.503},
    {"name": "Andrew Nembhard", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.469, "underPct": 0.531},
    {"name": "T.J. McConnell", "line": 3.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.446, "underPct": 0.554},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Scottie Barnes", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.429, "underPct": 0.571},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.406, "underPct": 0.594},
    {"name": "Collin Gillespie", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.374, "underPct": 0.626},
    {"name": "Jeremiah Fears", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.345, "underPct": 0.655},
    {"name": "Naji Marshall", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.33, "underPct": 0.67},
    {"name": "Jaylen Brown", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.328, "underPct": 0.672},
    {"name": "Aaron Gordon", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.303, "underPct": 0.697},
    {"name": "Stephen Curry", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.263, "underPct": 0.737},
];const underdogReboundsHitRates = [
    {"name": "Tre Jones", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.801, "underPct": 0.199},
    {"name": "Jamal Murray", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.744, "underPct": 0.256},
    {"name": "Donovan Mitchell", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.739, "underPct": 0.261},
    {"name": "Max Christie", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.698, "underPct": 0.302},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.682, "underPct": 0.318},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.669, "underPct": 0.331},
    {"name": "Alperen Sengun", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
    {"name": "Jalen Smith", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.568, "underPct": 0.432},
    {"name": "Luguentz Dort", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.549, "underPct": 0.451},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.524, "underPct": 0.476},
    {"name": "Amen Thompson", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.521, "underPct": 0.479},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.52, "underPct": 0.48},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.517, "underPct": 0.483},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.482, "underPct": 0.518},
    {"name": "Derik Queen", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.473, "underPct": 0.527},
    {"name": "Isaiah Jackson", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.472, "underPct": 0.528},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.459, "underPct": 0.541},
    {"name": "Chet Holmgren", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.438, "underPct": 0.562},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.436, "underPct": 0.564},
    {"name": "Ayo Dosunmu", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.421, "underPct": 0.579},
    {"name": "Jakob Poeltl", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.406, "underPct": 0.594},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Norman Powell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.398, "underPct": 0.602},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.396, "underPct": 0.604},
    {"name": "Deni Avdija", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.383, "underPct": 0.617},
    {"name": "Moses Moody", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.376, "underPct": 0.624},
    {"name": "Yves Missi", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.373, "underPct": 0.627},
    {"name": "Kevin Durant", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.369, "underPct": 0.631},
    {"name": "Luka Garza", "line": 5.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.368, "underPct": 0.632},
    {"name": "Will Richard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.342, "underPct": 0.658},
    {"name": "Klay Thompson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.335, "underPct": 0.665},
    {"name": "Dereck Lively II", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.326, "underPct": 0.674},
    {"name": "T.J. McConnell", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.323, "underPct": 0.677},
    {"name": "Draymond Green", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.292, "underPct": 0.708},
];const underdogBlocksHitRates = [
    {"name": "Scottie Barnes", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.614, "underPct": 0.386},
    {"name": "Daniel Gafford", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.487, "underPct": 0.513},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.49, "underPct": 0.51},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.507, "underPct": 0.493},
];const underdogStealsHitRates = [
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.531, "underPct": 0.469},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.572, "underPct": 0.428},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Kel'el Ware", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylon Tyson", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 40.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Day'Ron Sharpe", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Jones", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jrue Holiday", "line": 26.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Derrick White", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 38.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Robinson-Earl", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Simone Fontecchio", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 29.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 37.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Oso Ighodaro", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 34.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Gordon", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Stephen Curry", "line": 36.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ace Bailey", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaac Okoro", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Whitmore", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anfernee Simons", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Huerter", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dean Wade", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alex Sarr", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Buddy Hield", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dru Smith", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Hartenstein", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 39.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Coby White", "line": 28.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ajay Mitchell", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Will Richard", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pelle Larsson", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 39.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dereck Lively II", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "P.J. Washington", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 22.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 28.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogPRHitRates = [
    {"name": "Kel'el Ware", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Lauri Markkanen", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jrue Holiday", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shaedon Sharpe", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 32.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Hartenstein", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Coby White", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bam Adebayo", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "P.J. Washington", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogPAHitRates = [
    {"name": "Shaedon Sharpe", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jrue Holiday", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Nembhard", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Darius Garland", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 36.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ajay Mitchell", "line": 19.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogRAHitRates = [
    {"name": "Kel'el Ware", "line": 10.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 17.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "D'Angelo Russell", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jrue Holiday", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dereck Lively II", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alex Sarr", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derrick White", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const underdogTurnoversHitRates = [
    {"name": "Jrue Holiday", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogBlocksStealsHitRates = [
    {"name": "Derik Queen", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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
            <th style="width: 2%">#</th>
            <th style="width: 14%">Player </th>
            <th style="width: 5%">Line </th>
            <th style="width: 5%">Proj. </th>
            <th style="width: 5%">Prob. </th>
            <th style="width: 14%">Player </th>
            <th style="width: 5%">Line </th>
            <th style="width: 5%">Proj. </th>
            <th style="width: 5%">Prob. </th>
            <th style="width: 8%">EV $</th>
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
            <th style="width: 7%">EV $</th>
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
                <div class="stat-label">Probability</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">How confident the model is that the player will hit their prop line.</div>
            </div>
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

