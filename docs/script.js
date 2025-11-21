const prizepicksSinglesData = [
    {"name": "Jerami Grant", "bookmaker": "DraftKings", "line": 22.5, "prediction": 17.1, "side": "Under", "odds": -111, "recommendation": 1, "ev": 6.09, "kelly": 0.676, "sigma": "Med"},
    {"name": "Tre Jones", "bookmaker": "BetMGM", "line": 9.5, "prediction": 14.89, "side": "Over", "odds": -110, "recommendation": 1, "ev": 5.71, "kelly": 0.628, "sigma": "Med"},
    {"name": "Alperen Sengun", "bookmaker": "BetRivers", "line": 24.5, "prediction": 28.14, "side": "Over", "odds": 112, "recommendation": 0, "ev": 5.09, "kelly": 0.455, "sigma": "High"},
    {"name": "Bennedict Mathurin", "bookmaker": "BetMGM", "line": 21.5, "prediction": 25.16, "side": "Over", "odds": 110, "recommendation": 0, "ev": 5.07, "kelly": 0.461, "sigma": "High"},
    {"name": "Evan Mobley", "bookmaker": "BetRivers", "line": 18.5, "prediction": 14.84, "side": "Under", "odds": 108, "recommendation": 0, "ev": 5.06, "kelly": 0.469, "sigma": "High"},
    {"name": "Jonas Valanciunas", "bookmaker": "BetMGM", "line": 6.5, "prediction": 8.83, "side": "Over", "odds": 105, "recommendation": 0, "ev": 4.6, "kelly": 0.439, "sigma": "Low"},
    {"name": "Lauri Markkanen", "bookmaker": "FanDuel", "line": 24.5, "prediction": 28.82, "side": "Over", "odds": -102, "recommendation": 1, "ev": 4.54, "kelly": 0.463, "sigma": "High"},
    {"name": "Isaac Okoro", "bookmaker": "BetMGM", "line": 7.5, "prediction": 11.22, "side": "Over", "odds": -105, "recommendation": 0, "ev": 4.32, "kelly": 0.453, "sigma": "Med"},
    {"name": "Saddiq Bey", "bookmaker": "DraftKings", "line": 8.5, "prediction": 11.87, "side": "Over", "odds": 102, "recommendation": 0, "ev": 4.15, "kelly": 0.407, "sigma": "High"},
    {"name": "Julius Randle", "bookmaker": "FanDuel", "line": 22.5, "prediction": 26.17, "side": "Over", "odds": 100, "recommendation": 0, "ev": 4.14, "kelly": 0.414, "sigma": "High"},
];const prizepicksPairsData = [
    {"name1": "Tre Jones", "name2": "Jerami Grant", "line1": 9.5, "line2": 22.5, "prediction1": 14.89, "prediction2": 17.1, "side1": "over", "side2": "under", "recommendation": 1, "ev": 10.48, "kelly": 0.524, "sigma1": "Med", "sigma2": "Med", "prob1": 0.823, "prob2": 0.847, "hitRate1": 85.0, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 82.4, "l5_2": 0.4, "l15_2": 0.2},
    {"name1": "Evan Mobley", "name2": "Alperen Sengun", "line1": 19.5, "line2": 23.5, "prediction1": 14.84, "prediction2": 28.14, "side1": "under", "side2": "over", "recommendation": 1, "ev": 7.38, "kelly": 0.369, "sigma1": "High", "sigma2": "High", "prob1": 0.776, "prob2": 0.762, "hitRate1": 61.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 53.2, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Bennedict Mathurin", "name2": "Julius Randle", "line1": 20.5, "line2": 21.5, "prediction1": 25.16, "prediction2": 26.17, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.07, "kelly": 0.354, "sigma1": "High", "sigma2": "High", "prob1": 0.768, "prob2": 0.756, "hitRate1": 56.9, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 71.3, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Dillon Brooks", "name2": "Lauri Markkanen", "line1": 18.5, "line2": 24.5, "prediction1": 22.8, "prediction2": 28.82, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.03, "kelly": 0.301, "sigma1": "High", "sigma2": "High", "prob1": 0.743, "prob2": 0.734, "hitRate1": 68.5, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 87.4, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Isaac Okoro", "name2": "Aaron Gordon", "line1": 7.5, "line2": 16.5, "prediction1": 11.22, "prediction2": 19.75, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.79, "kelly": 0.289, "sigma1": "Med", "sigma2": "Med", "prob1": 0.733, "prob2": 0.732, "hitRate1": 65.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 76.9, "l5_2": 1.0, "l15_2": 0.6},
    {"name1": "Cameron Johnson", "name2": "Buddy Hield", "line1": 12.5, "line2": 7.5, "prediction1": 9.26, "prediction2": 10.83, "side1": "under", "side2": "over", "recommendation": 0, "ev": 5.42, "kelly": 0.271, "sigma1": "Med", "sigma2": "Med", "prob1": 0.724, "prob2": 0.724, "hitRate1": 88.1, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 36.6, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Pascal Siakam", "name2": "Saddiq Bey", "line1": 22.5, "line2": 8.5, "prediction1": 25.36, "prediction2": 11.87, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.7, "kelly": 0.185, "sigma1": "High", "sigma2": "High", "prob1": 0.665, "prob2": 0.7, "hitRate1": 46.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 91.0, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Khris Middleton", "name2": "D'Angelo Russell", "line1": 10.5, "line2": 12.5, "prediction1": 12.73, "prediction2": 15.66, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.32, "kelly": 0.166, "sigma1": "Med", "sigma2": "High", "prob1": 0.663, "prob2": 0.684, "hitRate1": 23.9, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 41.0, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Andrew Nembhard", "name2": "Cooper Flagg", "line1": 15.5, "line2": 17.5, "prediction1": 18.42, "prediction2": 14.46, "side1": "over", "side2": "under", "recommendation": 0, "ev": 3.3, "kelly": 0.165, "sigma1": "High", "sigma2": "High", "prob1": 0.663, "prob2": 0.683, "hitRate1": 77.7, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 55.2, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Jeremiah Fears", "name2": "Keyonte George", "line1": 14.5, "line2": 18.5, "prediction1": 17.24, "prediction2": 21.23, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.63, "kelly": 0.131, "sigma1": "High", "sigma2": "High", "prob1": 0.656, "prob2": 0.654, "hitRate1": 72.2, "l5_1": 1.0, "l15_1": 0.67, "hitRate2": 84.1, "l5_2": 0.8, "l15_2": 0.6},
];const prizepicksTriosData = [
    {"name1": "Evan Mobley", "name2": "Tre Jones", "name3": "Jerami Grant", "line1": 19.5, "line2": 9.5, "line3": 22.5, "prediction1": 14.84, "prediction2": 14.89, "prediction3": 17.1, "side1": "under", "side2": "over", "side3": "under", "recommendation": 1, "ev": 19.19, "kelly": 0.384, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.776, "prob2": 0.823, "prob3": 0.847, "hitRate1": 61.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 85.0, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 82.4, "l5_3": 0.4, "l15_3": 0.2},
    {"name1": "Bennedict Mathurin", "name2": "Julius Randle", "name3": "Alperen Sengun", "line1": 20.5, "line2": 21.5, "line3": 23.5, "prediction1": 25.16, "prediction2": 26.17, "prediction3": 28.14, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 13.9, "kelly": 0.278, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.768, "prob2": 0.756, "prob3": 0.762, "hitRate1": 56.9, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 71.3, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 53.2, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Isaac Okoro", "name2": "Dillon Brooks", "name3": "Lauri Markkanen", "line1": 7.5, "line2": 18.5, "line3": 24.5, "prediction1": 11.22, "prediction2": 22.8, "prediction3": 28.82, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.58, "kelly": 0.232, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.733, "prob2": 0.743, "prob3": 0.734, "hitRate1": 65.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 68.5, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 87.4, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Aaron Gordon", "name2": "Cameron Johnson", "name3": "Buddy Hield", "line1": 16.5, "line2": 12.5, "line3": 7.5, "prediction1": 19.75, "prediction2": 9.26, "prediction3": 10.83, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 10.74, "kelly": 0.215, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "prob1": 0.732, "prob2": 0.724, "prob3": 0.724, "hitRate1": 76.9, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 88.1, "l5_2": 0.4, "l15_2": 0.2, "hitRate3": 36.6, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Pascal Siakam", "name2": "Cooper Flagg", "name3": "D'Angelo Russell", "line1": 22.5, "line2": 17.5, "line3": 12.5, "prediction1": 25.36, "prediction2": 14.46, "prediction3": 15.66, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 6.78, "kelly": 0.136, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.665, "prob2": 0.683, "prob3": 0.684, "hitRate1": 46.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 55.2, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 41.0, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Andrew Nembhard", "name2": "Khris Middleton", "name3": "Saddiq Bey", "line1": 15.5, "line2": 10.5, "line3": 8.5, "prediction1": 18.42, "prediction2": 12.73, "prediction3": 11.87, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.6, "kelly": 0.132, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.663, "prob2": 0.663, "prob3": 0.7, "hitRate1": 77.7, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 23.9, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 91.0, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Jeremiah Fears", "name2": "Nikola Joki\u0107", "name3": "Keyonte George", "line1": 14.5, "line2": 28.5, "line3": 18.5, "prediction1": 17.24, "prediction2": 30.83, "prediction3": 21.23, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.15, "kelly": 0.103, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.656, "prob2": 0.653, "prob3": 0.654, "hitRate1": 72.2, "l5_1": 1.0, "l15_1": 0.67, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 84.1, "l5_3": 0.8, "l15_3": 0.6},
    {"name1": "Matas Buzelis", "name2": "Naji Marshall", "name3": "Will Richard", "line1": 14.5, "line2": 11.5, "line3": 7.5, "prediction1": 17.2, "prediction2": 13.91, "prediction3": 9.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.04, "kelly": 0.101, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.652, "prob2": 0.655, "prob3": 0.653, "hitRate1": 59.6, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 82.6, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 43.3, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Isaiah Jackson", "name2": "Brandon Ingram", "name3": "Shai Gilgeous-Alexander", "line1": 7.5, "line2": 21.5, "line3": 31.5, "prediction1": 9.24, "prediction2": 23.71, "prediction3": 29.41, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 3.98, "kelly": 0.08, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.633, "prob2": 0.635, "prob3": 0.644, "hitRate1": 62.2, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 45.7, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 63.4, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Tre Johnson", "name2": "Simone Fontecchio", "name3": "Zion Williamson", "line1": 8.5, "line2": 10.5, "line3": 23.5, "prediction1": 10.51, "prediction2": 12.38, "prediction3": 21.41, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 3.47, "kelly": 0.069, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.629, "prob2": 0.63, "prob3": 0.63, "hitRate1": 68.9, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 53.7, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 65.7, "l5_3": 0.4, "l15_3": 0.2},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Tre Jones", "name2": "Jerami Grant", "line1": 9.5, "line2": 22.5, "prediction1": 14.89, "prediction2": 17.1, "side1": "over", "side2": "under", "recommendation": 1, "ev": 10.48, "kelly": 0.524, "sigma1": "Med", "sigma2": "Med", "prob1": 0.823, "prob2": 0.847, "hitRate1": 85.0, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 82.4, "l5_2": 0.4, "l15_2": 0.2},
    {"name1": "Evan Mobley", "name2": "Alperen Sengun", "line1": 19.5, "line2": 23.5, "prediction1": 14.84, "prediction2": 28.14, "side1": "under", "side2": "over", "recommendation": 1, "ev": 7.38, "kelly": 0.369, "sigma1": "High", "sigma2": "High", "prob1": 0.776, "prob2": 0.762, "hitRate1": 61.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 53.2, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Bennedict Mathurin", "name2": "Dillon Brooks", "line1": 20.5, "line2": 18.5, "prediction1": 25.16, "prediction2": 22.8, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.77, "kelly": 0.339, "sigma1": "High", "sigma2": "High", "prob1": 0.768, "prob2": 0.743, "hitRate1": 56.9, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 68.5, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Isaac Okoro", "name2": "Lauri Markkanen", "line1": 7.5, "line2": 24.5, "prediction1": 11.22, "prediction2": 28.82, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.82, "kelly": 0.291, "sigma1": "Med", "sigma2": "High", "prob1": 0.733, "prob2": 0.734, "hitRate1": 65.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 87.4, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Julius Randle", "name2": "Buddy Hield", "line1": 22.5, "line2": 7.5, "prediction1": 26.17, "prediction2": 10.83, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.05, "kelly": 0.253, "sigma1": "High", "sigma2": "Med", "prob1": 0.707, "prob2": 0.724, "hitRate1": 63.8, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 36.6, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Pascal Siakam", "name2": "Saddiq Bey", "line1": 22.5, "line2": 8.5, "prediction1": 25.36, "prediction2": 11.87, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.7, "kelly": 0.185, "sigma1": "High", "sigma2": "High", "prob1": 0.665, "prob2": 0.7, "hitRate1": 46.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 91.0, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Andrew Nembhard", "name2": "D'Angelo Russell", "line1": 15.5, "line2": 12.5, "prediction1": 18.42, "prediction2": 15.66, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.32, "kelly": 0.166, "sigma1": "High", "sigma2": "High", "prob1": 0.663, "prob2": 0.684, "hitRate1": 77.7, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 41.0, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Khris Middleton", "name2": "Cooper Flagg", "line1": 10.5, "line2": 17.5, "prediction1": 12.73, "prediction2": 14.46, "side1": "over", "side2": "under", "recommendation": 0, "ev": 3.3, "kelly": 0.165, "sigma1": "Med", "sigma2": "High", "prob1": 0.663, "prob2": 0.683, "hitRate1": 23.9, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 55.2, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Jeremiah Fears", "name2": "Keyonte George", "line1": 14.5, "line2": 18.5, "prediction1": 17.24, "prediction2": 21.23, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.63, "kelly": 0.131, "sigma1": "High", "sigma2": "High", "prob1": 0.656, "prob2": 0.654, "hitRate1": 72.2, "l5_1": 1.0, "l15_1": 0.67, "hitRate2": 84.1, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Naji Marshall", "name2": "Nikola Joki\u0107", "line1": 11.5, "line2": 28.5, "prediction1": 13.91, "prediction2": 30.83, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.57, "kelly": 0.129, "sigma1": "High", "sigma2": "Med", "prob1": 0.655, "prob2": 0.653, "hitRate1": 82.6, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
];const underdogTriosData = [
    {"name1": "Evan Mobley", "name2": "Tre Jones", "name3": "Jerami Grant", "line1": 19.5, "line2": 9.5, "line3": 22.5, "prediction1": 14.84, "prediction2": 14.89, "prediction3": 17.1, "side1": "under", "side2": "over", "side3": "under", "recommendation": 1, "ev": 19.19, "kelly": 0.384, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.776, "prob2": 0.823, "prob3": 0.847, "hitRate1": 61.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 85.0, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 82.4, "l5_3": 0.4, "l15_3": 0.2},
    {"name1": "Bennedict Mathurin", "name2": "Dillon Brooks", "name3": "Alperen Sengun", "line1": 20.5, "line2": 18.5, "line3": 23.5, "prediction1": 25.16, "prediction2": 22.8, "prediction3": 28.14, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 13.48, "kelly": 0.27, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.768, "prob2": 0.743, "prob3": 0.762, "hitRate1": 56.9, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 68.5, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 53.2, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Isaac Okoro", "name2": "Buddy Hield", "name3": "Lauri Markkanen", "line1": 7.5, "line2": 7.5, "line3": 24.5, "prediction1": 11.22, "prediction2": 10.83, "prediction3": 28.82, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.05, "kelly": 0.221, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.733, "prob2": 0.724, "prob3": 0.734, "hitRate1": 65.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 36.6, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 87.4, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Cooper Flagg", "name2": "D'Angelo Russell", "name3": "Julius Randle", "line1": 17.5, "line2": 12.5, "line3": 22.5, "prediction1": 14.46, "prediction2": 15.66, "prediction3": 26.17, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.83, "kelly": 0.157, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.683, "prob2": 0.684, "prob3": 0.707, "hitRate1": 55.2, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 41.0, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 63.8, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Pascal Siakam", "name2": "Andrew Nembhard", "name3": "Saddiq Bey", "line1": 22.5, "line2": 15.5, "line3": 8.5, "prediction1": 25.36, "prediction2": 18.42, "prediction3": 11.87, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.68, "kelly": 0.134, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.665, "prob2": 0.663, "prob3": 0.7, "hitRate1": 46.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 77.7, "l5_2": 0.8, "l15_2": 0.33, "hitRate3": 91.0, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Khris Middleton", "name2": "Jeremiah Fears", "name3": "Keyonte George", "line1": 10.5, "line2": 14.5, "line3": 18.5, "prediction1": 12.73, "prediction2": 17.24, "prediction3": 21.23, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.37, "kelly": 0.107, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.663, "prob2": 0.656, "prob3": 0.654, "hitRate1": 23.9, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 72.2, "l5_2": 1.0, "l15_2": 0.67, "hitRate3": 84.1, "l5_3": 0.8, "l15_3": 0.6},
    {"name1": "Naji Marshall", "name2": "Nikola Joki\u0107", "name3": "Will Richard", "line1": 11.5, "line2": 28.5, "line3": 7.5, "prediction1": 13.91, "prediction2": 30.83, "prediction3": 9.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.07, "kelly": 0.101, "sigma1": "High", "sigma2": "Med", "sigma3": "Low", "prob1": 0.655, "prob2": 0.653, "prob3": 0.653, "hitRate1": 82.6, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 43.3, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Donovan Mitchell", "name2": "Matas Buzelis", "name3": "Shai Gilgeous-Alexander", "line1": 28.5, "line2": 14.5, "line3": 31.5, "prediction1": 26.01, "prediction2": 17.2, "prediction3": 29.41, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 4.53, "kelly": 0.091, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.641, "prob2": 0.652, "prob3": 0.644, "hitRate1": 48.1, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 59.6, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 63.4, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Isaiah Jackson", "name2": "Zion Williamson", "name3": "Ace Bailey", "line1": 7.5, "line2": 23.5, "line3": 10.5, "prediction1": 9.24, "prediction2": 21.41, "prediction3": 12.53, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 3.77, "kelly": 0.075, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.633, "prob2": 0.63, "prob3": 0.64, "hitRate1": 62.2, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 65.7, "l5_2": 0.4, "l15_2": 0.2, "hitRate3": 66.9, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Simone Fontecchio", "name2": "Derik Queen", "name3": "Devin Booker", "line1": 10.5, "line2": 13.5, "line3": 28.5, "prediction1": 12.38, "prediction2": 15.49, "prediction3": 30.64, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 3.17, "kelly": 0.063, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.63, "prob2": 0.622, "prob3": 0.623, "hitRate1": 53.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 61.4, "l5_2": 0.4, "l15_2": 0.2, "hitRate3": 25.1, "l5_3": 0.2, "l15_3": 0.47},
];const prizepicksPointsHitRates = [
    {"name": "Saddiq Bey", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.91, "underPct": 0.09},
    {"name": "Lauri Markkanen", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.874, "underPct": 0.126},
    {"name": "Trey Murphy III", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.854, "underPct": 0.146},
    {"name": "Tre Jones", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.85, "underPct": 0.15},
    {"name": "Keyonte George", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.841, "underPct": 0.159},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.826, "underPct": 0.174},
    {"name": "Andrew Nembhard", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.777, "underPct": 0.223},
    {"name": "Aaron Gordon", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.769, "underPct": 0.231},
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.737, "underPct": 0.263},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.722, "underPct": 0.278},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.716, "underPct": 0.284},
    {"name": "Julius Randle", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.713, "underPct": 0.287},
    {"name": "Kyle Filipowski", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.701, "underPct": 0.299},
    {"name": "Tre Johnson", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.689, "underPct": 0.311},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.685, "underPct": 0.315},
    {"name": "Sandro Mamukelashvili", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.684, "underPct": 0.316},
    {"name": "Reed Sheppard", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.678, "underPct": 0.322},
    {"name": "Noah Clowney", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.674, "underPct": 0.326},
    {"name": "Ayo Dosunmu", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.672, "underPct": 0.328},
    {"name": "Stephen Curry", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.666, "underPct": 0.334},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.665, "underPct": 0.335},
    {"name": "Naz Reid", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.661, "underPct": 0.339},
    {"name": "Josh Minott", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.654, "underPct": 0.346},
    {"name": "Isaac Okoro", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.654, "underPct": 0.346},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.647, "underPct": 0.353},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.637, "underPct": 0.363},
    {"name": "Payton Pritchard", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.627, "underPct": 0.373},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.622, "underPct": 0.378},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.614, "underPct": 0.386},
    {"name": "De'Andre Hunter", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.61, "underPct": 0.39},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.607, "underPct": 0.393},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.605, "underPct": 0.395},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.596, "underPct": 0.404},
    {"name": "Day'Ron Sharpe", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.576, "underPct": 0.424},
    {"name": "Jamal Murray", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.573, "underPct": 0.427},
    {"name": "Mark Williams", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.572, "underPct": 0.428},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.572, "underPct": 0.428},
    {"name": "Isaiah Joe", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.57, "underPct": 0.43},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.569, "underPct": 0.431},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.569, "underPct": 0.431},
    {"name": "Ace Bailey", "line": 11.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.556, "underPct": 0.444},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.556, "underPct": 0.444},
    {"name": "Jakob Poeltl", "line": 14.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jordan Goodwin", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.55, "underPct": 0.45},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.54, "underPct": 0.46},
    {"name": "Mike Conley", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.538, "underPct": 0.462},
    {"name": "Simone Fontecchio", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.537, "underPct": 0.463},
    {"name": "Peyton Watson", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.535, "underPct": 0.465},
    {"name": "Alperen Sengun", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.532, "underPct": 0.468},
    {"name": "Jeremiah Robinson-Earl", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.528, "underPct": 0.472},
    {"name": "Corey Kispert", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.527, "underPct": 0.473},
    {"name": "Cam Whitmore", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.524, "underPct": 0.476},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.523, "underPct": 0.477},
    {"name": "Donte DiVincenzo", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.518, "underPct": 0.482},
    {"name": "Lonzo Ball", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.509, "underPct": 0.491},
    {"name": "Dru Smith", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.503, "underPct": 0.497},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.499, "underPct": 0.501},
    {"name": "Brice Sensabaugh", "line": 8.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.497, "underPct": 0.503},
    {"name": "Tyrese Martin", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.492, "underPct": 0.508},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.488, "underPct": 0.512},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.478, "underPct": 0.522},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.475, "underPct": 0.525},
    {"name": "Anthony Edwards", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.474, "underPct": 0.526},
    {"name": "Pascal Siakam", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.467, "underPct": 0.533},
    {"name": "Josh Giddey", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.464, "underPct": 0.536},
    {"name": "Brandin Podziemski", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.463, "underPct": 0.537},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.463, "underPct": 0.537},
    {"name": "Jarace Walker", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.461, "underPct": 0.539},
    {"name": "Brandon Ingram", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.457, "underPct": 0.543},
    {"name": "Deni Avdija", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.454, "underPct": 0.546},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.452, "underPct": 0.548},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.448, "underPct": 0.552},
    {"name": "Luguentz Dort", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.441, "underPct": 0.559},
    {"name": "Will Richard", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.433, "underPct": 0.567},
    {"name": "Moses Moody", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.431, "underPct": 0.569},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Anfernee Simons", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.422, "underPct": 0.578},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.41, "underPct": 0.59},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.401, "underPct": 0.599},
    {"name": "Ajay Mitchell", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.396, "underPct": 0.604},
    {"name": "Bam Adebayo", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.391, "underPct": 0.609},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.381, "underPct": 0.619},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.366, "underPct": 0.634},
    {"name": "Buddy Hield", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.366, "underPct": 0.634},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.362, "underPct": 0.638},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.351, "underPct": 0.649},
    {"name": "Zion Williamson", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.343, "underPct": 0.657},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.336, "underPct": 0.664},
    {"name": "Bilal Coulibaly", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.326, "underPct": 0.674},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.316, "underPct": 0.684},
    {"name": "Bruce Brown", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.316, "underPct": 0.684},
    {"name": "Max Christie", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.275, "underPct": 0.725},
    {"name": "Jordan Hawkins", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.256, "underPct": 0.744},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.251, "underPct": 0.749},
    {"name": "Collin Gillespie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.244, "underPct": 0.756},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.24, "underPct": 0.76},
    {"name": "Khris Middleton", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.239, "underPct": 0.761},
    {"name": "Terance Mann", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.235, "underPct": 0.765},
    {"name": "Ryan Dunn", "line": 9.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.22, "underPct": 0.78},
    {"name": "Al Horford", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.218, "underPct": 0.782},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.215, "underPct": 0.785},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.203, "underPct": 0.797},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.183, "underPct": 0.817},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.176, "underPct": 0.824},
    {"name": "Dereck Lively II", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.143, "underPct": 0.857},
    {"name": "Cameron Johnson", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.119, "underPct": 0.881},
];const prizepicksAssistsHitRates = [
    {"name": "Isaiah Collier", "line": 5.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.726, "underPct": 0.274},
    {"name": "Josh Giddey", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.693, "underPct": 0.307},
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.632, "underPct": 0.368},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.632, "underPct": 0.368},
    {"name": "Derrick White", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.58, "underPct": 0.42},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.548, "underPct": 0.452},
    {"name": "Kel'el Ware", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.505, "underPct": 0.495},
    {"name": "Alperen Sengun", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.498, "underPct": 0.502},
    {"name": "Zion Williamson", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.496, "underPct": 0.504},
    {"name": "Jamal Murray", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.49, "underPct": 0.51},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Jaden McDaniels", "line": 2.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
    {"name": "Pascal Siakam", "line": 4.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.445, "underPct": 0.555},
    {"name": "T.J. McConnell", "line": 3.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.438, "underPct": 0.562},
    {"name": "Draymond Green", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.401, "underPct": 0.599},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.397, "underPct": 0.603},
    {"name": "Amen Thompson", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.394, "underPct": 0.606},
    {"name": "Devin Booker", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.386, "underPct": 0.614},
    {"name": "Immanuel Quickley", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.386, "underPct": 0.614},
    {"name": "D'Angelo Russell", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.386, "underPct": 0.614},
    {"name": "Davion Mitchell", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.364, "underPct": 0.636},
    {"name": "Ajay Mitchell", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.359, "underPct": 0.641},
    {"name": "Collin Gillespie", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.357, "underPct": 0.643},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.355, "underPct": 0.645},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.35, "underPct": 0.65},
    {"name": "Payton Pritchard", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.333, "underPct": 0.667},
    {"name": "Shai Gilgeous-Alexander", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.312, "underPct": 0.688},
    {"name": "Jamal Shead", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.304, "underPct": 0.696},
    {"name": "Bruce Brown", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.295, "underPct": 0.705},
    {"name": "Kris Murray", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.292, "underPct": 0.708},
    {"name": "Stephen Curry", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.273, "underPct": 0.727},
];const prizepicksReboundsHitRates = [
    {"name": "Saddiq Bey", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.811, "underPct": 0.189},
    {"name": "Kel'el Ware", "line": 9.0, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.79, "underPct": 0.21},
    {"name": "Jamal Murray", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.752, "underPct": 0.248},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.69, "underPct": 0.31},
    {"name": "Naji Marshall", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.673, "underPct": 0.327},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.658, "underPct": 0.342},
    {"name": "Matas Buzelis", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.642, "underPct": 0.358},
    {"name": "Alperen Sengun", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.605, "underPct": 0.395},
    {"name": "Isaiah Joe", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.572, "underPct": 0.428},
    {"name": "Bruce Brown", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.562, "underPct": 0.438},
    {"name": "Luguentz Dort", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.558, "underPct": 0.442},
    {"name": "Jalen Smith", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.545, "underPct": 0.455},
    {"name": "Scottie Barnes", "line": 8.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.541, "underPct": 0.459},
    {"name": "Amen Thompson", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.539, "underPct": 0.461},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.538, "underPct": 0.462},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.521, "underPct": 0.479},
    {"name": "Zion Williamson", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.517, "underPct": 0.483},
    {"name": "Isaiah Hartenstein", "line": 10.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.511, "underPct": 0.489},
    {"name": "Jay Huff", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.502, "underPct": 0.498},
    {"name": "Payton Pritchard", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.493, "underPct": 0.507},
    {"name": "Immanuel Quickley", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.493, "underPct": 0.507},
    {"name": "Jaylen Brown", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.492, "underPct": 0.508},
    {"name": "Naz Reid", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.491, "underPct": 0.509},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.478, "underPct": 0.522},
    {"name": "Devin Booker", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.468, "underPct": 0.532},
    {"name": "Shai Gilgeous-Alexander", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.467, "underPct": 0.533},
    {"name": "Derrick White", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.464, "underPct": 0.536},
    {"name": "Aaron Gordon", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.46, "underPct": 0.54},
    {"name": "Collin Gillespie", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.459, "underPct": 0.541},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.456, "underPct": 0.544},
    {"name": "Cooper Flagg", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.455, "underPct": 0.545},
    {"name": "Bilal Coulibaly", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.453, "underPct": 0.547},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.445, "underPct": 0.555},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.441, "underPct": 0.559},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.43, "underPct": 0.57},
    {"name": "Donovan Clingan", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.42, "underPct": 0.58},
    {"name": "P.J. Washington", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.419, "underPct": 0.581},
    {"name": "Royce O'Neale", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.418, "underPct": 0.582},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.415, "underPct": 0.585},
    {"name": "Norman Powell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.41, "underPct": 0.59},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.39, "underPct": 0.61},
    {"name": "Jarace Walker", "line": 4.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.387, "underPct": 0.613},
    {"name": "Kevin Durant", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "Dillon Brooks", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.383, "underPct": 0.617},
    {"name": "Kris Murray", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.38, "underPct": 0.62},
    {"name": "Mark Williams", "line": 8.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.369, "underPct": 0.631},
    {"name": "Isaiah Jackson", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.341, "underPct": 0.659},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.337, "underPct": 0.663},
    {"name": "Will Richard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.332, "underPct": 0.668},
    {"name": "Josh Minott", "line": 4.0, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.33, "underPct": 0.67},
    {"name": "Pascal Siakam", "line": 6.0, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.321, "underPct": 0.679},
    {"name": "Klay Thompson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.318, "underPct": 0.682},
    {"name": "T.J. McConnell", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.316, "underPct": 0.684},
    {"name": "Dereck Lively II", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.315, "underPct": 0.685},
    {"name": "Daniel Gafford", "line": 7.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.312, "underPct": 0.688},
    {"name": "Draymond Green", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.304, "underPct": 0.696},
    {"name": "Lauri Markkanen", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.303, "underPct": 0.697},
    {"name": "Al Horford", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.267, "underPct": 0.733},
    {"name": "Sandro Mamukelashvili", "line": 5.0, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.249, "underPct": 0.751},
    {"name": "Stephen Curry", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.234, "underPct": 0.766},
    {"name": "Khris Middleton", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.229, "underPct": 0.771},
    {"name": "Noah Clowney", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.225, "underPct": 0.775},
    {"name": "Kyshawn George", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.224, "underPct": 0.776},
    {"name": "Ryan Dunn", "line": 5.0, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.221, "underPct": 0.779},
    {"name": "Pelle Larsson", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.21, "underPct": 0.79},
];const prizepicksBlocksHitRates = [
    {"name": "Brandon Ingram", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.59, "underPct": 0.41},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.543, "underPct": 0.457},
    {"name": "Cooper Flagg", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.567, "underPct": 0.433},
    {"name": "Zion Williamson", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.281, "underPct": 0.719},
    {"name": "Rudy Gobert", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.588, "underPct": 0.412},
    {"name": "Kevin Durant", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.513, "underPct": 0.487},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.517, "underPct": 0.483},
];const prizepicksStealsHitRates = [
    {"name": "Jarace Walker", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.438, "underPct": 0.562},
    {"name": "Day'Ron Sharpe", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.586, "underPct": 0.414},
    {"name": "Neemias Queta", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.457, "underPct": 0.543},
    {"name": "Terance Mann", "line": 0.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.492, "underPct": 0.508},
    {"name": "Josh Minott", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.398, "underPct": 0.602},
    {"name": "Saddiq Bey", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.449, "underPct": 0.551},
    {"name": "Mike Conley", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.514, "underPct": 0.486},
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.539, "underPct": 0.461},
    {"name": "Donovan Clingan", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.411, "underPct": 0.589},
    {"name": "Al Horford", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.616, "underPct": 0.384},
    {"name": "Ace Bailey", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.566, "underPct": 0.434},
    {"name": "Isaiah Collier", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.28, "underPct": 0.72},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Kel'el Ware", "line": 21.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Jones", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Whitmore", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alperen Sengun", "line": 40.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Walsh", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Day'Ron Sharpe", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Filipowski", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Simone Fontecchio", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 16.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Robinson-Earl", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 37.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Gordon", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Oso Ighodaro", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 34.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 34.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bruce Brown", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 36.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Joe", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Davion Mitchell", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lonzo Ball", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaac Okoro", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ayo Dosunmu", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Martin", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 25.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 29.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Hartenstein", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 29.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Clingan", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandin Podziemski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Clark", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Khris Middleton", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 34.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sandro Mamukelashvili", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Dunn", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Buddy Hield", "line": 12.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ben Sheppard", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Dereck Lively II", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Will Richard", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 42.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pelle Larsson", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 40.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 29.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPRHitRates = [
    {"name": "Kel'el Ware", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Isaiah Collier", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ziaire Williams", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Goodwin", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 11.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Bennedict Mathurin", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Garza", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Reed Sheppard", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Derrick White", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tre Jones", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyle Filipowski", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 16.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaac Okoro", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Simone Fontecchio", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Joe", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tre Johnson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Robinson-Earl", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 22.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Whitmore", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Norman Powell", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Murray", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Peyton Watson", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anfernee Simons", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zion Williamson", "line": 29.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dru Smith", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ben Sheppard", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Hartenstein", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Jackson", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bilal Coulibaly", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "D'Angelo Russell", "line": 15.0, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dereck Lively II", "line": 13.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sandro Mamukelashvili", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 11.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jose Alvarado", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drake Powell", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Day'Ron Sharpe", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Minott", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Shead", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pelle Larsson", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 14.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Brown", "line": 33.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 21.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan Vukcevic", "line": 18.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPAHitRates = [
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Davion Mitchell", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Jones", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Saddiq Bey", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaden McDaniels", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jordan Goodwin", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Al Horford", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Chet Holmgren", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 13.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Filipowski", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luka Garza", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mike Conley", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Oso Ighodaro", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 21.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Joe", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Buddy Hield", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandin Podziemski", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Minott", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bam Adebayo", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pelle Larsson", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zion Williamson", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Daniel Gafford", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dereck Lively II", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Scottie Barnes", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 35.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drake Powell", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jarace Walker", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dru Smith", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ajay Mitchell", "line": 19.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const prizepicksRAHitRates = [
    {"name": "Alperen Sengun", "line": 17.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 13.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylon Tyson", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dillon Brooks", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 8.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden McDaniels", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Gordon", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 14.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 12.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 13.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pelle Larsson", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Chet Holmgren", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Simone Fontecchio", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Norman Powell", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sandro Mamukelashvili", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Peyton Watson", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 8.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 9.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Khris Middleton", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 12.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const prizepicksTurnoversHitRates = [
    {"name": "Anfernee Simons", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derrick White", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Hartenstein", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Sandro Mamukelashvili", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyle Filipowski", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Collier", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Minott", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Gordon", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Reed Sheppard", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dru Smith", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tristan Vukcevic", "line": 1.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const underdogPointsHitRates = [
    {"name": "Saddiq Bey", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.91, "underPct": 0.09},
    {"name": "Lauri Markkanen", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.874, "underPct": 0.126},
    {"name": "Jaylon Tyson", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.868, "underPct": 0.132},
    {"name": "Trey Murphy III", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.854, "underPct": 0.146},
    {"name": "Tre Jones", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.85, "underPct": 0.15},
    {"name": "Keyonte George", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.841, "underPct": 0.159},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.826, "underPct": 0.174},
    {"name": "Andrew Nembhard", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.777, "underPct": 0.223},
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.737, "underPct": 0.263},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.722, "underPct": 0.278},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.685, "underPct": 0.315},
    {"name": "Jalen Smith", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.68, "underPct": 0.32},
    {"name": "Reed Sheppard", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.678, "underPct": 0.322},
    {"name": "Noah Clowney", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.674, "underPct": 0.326},
    {"name": "Ayo Dosunmu", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.672, "underPct": 0.328},
    {"name": "Ace Bailey", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.669, "underPct": 0.331},
    {"name": "Stephen Curry", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.666, "underPct": 0.334},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.665, "underPct": 0.335},
    {"name": "Naz Reid", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.661, "underPct": 0.339},
    {"name": "Peyton Watson", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.655, "underPct": 0.345},
    {"name": "Isaac Okoro", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.654, "underPct": 0.346},
    {"name": "Josh Minott", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.654, "underPct": 0.346},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.647, "underPct": 0.353},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.638, "underPct": 0.362},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.637, "underPct": 0.363},
    {"name": "Payton Pritchard", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.627, "underPct": 0.373},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.622, "underPct": 0.378},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.614, "underPct": 0.386},
    {"name": "De'Andre Hunter", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.61, "underPct": 0.39},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.607, "underPct": 0.393},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.596, "underPct": 0.404},
    {"name": "Jamal Murray", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.573, "underPct": 0.427},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.572, "underPct": 0.428},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.569, "underPct": 0.431},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.569, "underPct": 0.431},
    {"name": "Darius Garland", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.553, "underPct": 0.447},
    {"name": "Jordan Goodwin", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.55, "underPct": 0.45},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.54, "underPct": 0.46},
    {"name": "Patrick Williams", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.539, "underPct": 0.461},
    {"name": "Mike Conley", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.538, "underPct": 0.462},
    {"name": "Simone Fontecchio", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.537, "underPct": 0.463},
    {"name": "Alperen Sengun", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.532, "underPct": 0.468},
    {"name": "Jeremiah Robinson-Earl", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.528, "underPct": 0.472},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.523, "underPct": 0.477},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.519, "underPct": 0.481},
    {"name": "Donte DiVincenzo", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.518, "underPct": 0.482},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.499, "underPct": 0.501},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.488, "underPct": 0.512},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.478, "underPct": 0.522},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.475, "underPct": 0.525},
    {"name": "Pascal Siakam", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.467, "underPct": 0.533},
    {"name": "Josh Giddey", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.464, "underPct": 0.536},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.463, "underPct": 0.537},
    {"name": "Bruce Brown", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.461, "underPct": 0.539},
    {"name": "Jarace Walker", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.461, "underPct": 0.539},
    {"name": "Deni Avdija", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.454, "underPct": 0.546},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.452, "underPct": 0.548},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.448, "underPct": 0.552},
    {"name": "Luguentz Dort", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.441, "underPct": 0.559},
    {"name": "Will Richard", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.433, "underPct": 0.567},
    {"name": "Moses Moody", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.431, "underPct": 0.569},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.41, "underPct": 0.59},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.401, "underPct": 0.599},
    {"name": "Ajay Mitchell", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.396, "underPct": 0.604},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.381, "underPct": 0.619},
    {"name": "Brandon Ingram", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.374, "underPct": 0.626},
    {"name": "Buddy Hield", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.366, "underPct": 0.634},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.366, "underPct": 0.634},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.362, "underPct": 0.638},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.356, "underPct": 0.644},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.351, "underPct": 0.649},
    {"name": "Zion Williamson", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.343, "underPct": 0.657},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.336, "underPct": 0.664},
    {"name": "Bilal Coulibaly", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.326, "underPct": 0.674},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.316, "underPct": 0.684},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.251, "underPct": 0.749},
    {"name": "Collin Gillespie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.244, "underPct": 0.756},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.24, "underPct": 0.76},
    {"name": "Khris Middleton", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.239, "underPct": 0.761},
    {"name": "Al Horford", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.218, "underPct": 0.782},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.215, "underPct": 0.785},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.203, "underPct": 0.797},
    {"name": "Dean Wade", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.195, "underPct": 0.805},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.183, "underPct": 0.817},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.176, "underPct": 0.824},
];const underdogAssistsHitRates = [
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.693, "underPct": 0.307},
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Jaylon Tyson", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.637, "underPct": 0.363},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.632, "underPct": 0.368},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.632, "underPct": 0.368},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.548, "underPct": 0.452},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Tre Jones", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.448, "underPct": 0.552},
    {"name": "T.J. McConnell", "line": 3.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.438, "underPct": 0.562},
    {"name": "Draymond Green", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.401, "underPct": 0.599},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.397, "underPct": 0.603},
    {"name": "Devin Booker", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.386, "underPct": 0.614},
    {"name": "Bruce Brown", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.295, "underPct": 0.705},
    {"name": "Kris Murray", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.292, "underPct": 0.708},
    {"name": "Luguentz Dort", "line": 1.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.242, "underPct": 0.758},
];const underdogReboundsHitRates = [
    {"name": "Jamal Murray", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.752, "underPct": 0.248},
    {"name": "Kyle Filipowski", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.705, "underPct": 0.295},
    {"name": "Max Christie", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.687, "underPct": 0.313},
    {"name": "Isaiah Collier", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.64, "underPct": 0.36},
    {"name": "Patrick Williams", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.635, "underPct": 0.365},
    {"name": "Isaac Okoro", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.612, "underPct": 0.388},
    {"name": "Alperen Sengun", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.605, "underPct": 0.395},
    {"name": "Isaiah Joe", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.572, "underPct": 0.428},
    {"name": "Bennedict Mathurin", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.566, "underPct": 0.434},
    {"name": "Amen Thompson", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.539, "underPct": 0.461},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.538, "underPct": 0.462},
    {"name": "Jaden McDaniels", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.525, "underPct": 0.475},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.511, "underPct": 0.489},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.493, "underPct": 0.507},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.478, "underPct": 0.522},
    {"name": "Peyton Watson", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.467, "underPct": 0.533},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.456, "underPct": 0.544},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.43, "underPct": 0.57},
    {"name": "Norman Powell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.41, "underPct": 0.59},
    {"name": "Ayo Dosunmu", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.408, "underPct": 0.592},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.39, "underPct": 0.61},
    {"name": "Kevin Durant", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "Klay Thompson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.318, "underPct": 0.682},
    {"name": "Andrew Nembhard", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.286, "underPct": 0.714},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.267, "underPct": 0.733},
];const underdogBlocksHitRates = [
    {"name": "Daniel Gafford", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.473, "underPct": 0.527},
    {"name": "Rudy Gobert", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.588, "underPct": 0.412},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.517, "underPct": 0.483},
];const underdogStealsHitRates = [
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.539, "underPct": 0.461},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.559, "underPct": 0.441},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Kel'el Ware", "line": 21.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 36.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Simone Fontecchio", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Jones", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Filipowski", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Day'Ron Sharpe", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 43.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 38.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Nembhard", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Robinson-Earl", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 40.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mike Conley", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donte DiVincenzo", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 37.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Al Horford", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 36.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 34.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Smith", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Payton Pritchard", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaac Okoro", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Patrick Williams", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Martin", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 29.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Moses Moody", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luguentz Dort", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Khris Middleton", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gradey Dick", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Hawkins", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bilal Coulibaly", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dereck Lively II", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "P.J. Washington", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ben Sheppard", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ajay Mitchell", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 22.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 43.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Peyton Watson", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 40.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pelle Larsson", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan Vukcevic", "line": 20.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Amen Thompson", "line": 29.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogPRHitRates = [
    {"name": "Kel'el Ware", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Derrick White", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Norman Powell", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 36.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogPAHitRates = [
    {"name": "Keyonte George", "line": 24.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Andrew Nembhard", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Andre Hunter", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bam Adebayo", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "P.J. Washington", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 36.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 31.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ajay Mitchell", "line": 19.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogRAHitRates = [
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tre Jones", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Smith", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dereck Lively II", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "P.J. Washington", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 7.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const underdogTurnoversHitRates = [
    {"name": "Josh Giddey", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogBlocksStealsHitRates = [
    {"name": "Derrick White", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
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

