const prizepicksSinglesData = [
    {"name": "Dillon Brooks", "bookmaker": "Bovada", "line": 20.5, "prediction": 22.52, "side": "Over", "odds": 205, "recommendation": 0, "ev": 8.53, "roi": 85.3, "kelly": 0.416, "sigma": "High"},
    {"name": "Dillon Brooks", "bookmaker": "Bovada", "line": 19.5, "prediction": 22.52, "side": "Over", "odds": 165, "recommendation": 0, "ev": 7.18, "roi": 71.8, "kelly": 0.435, "sigma": "High"},
    {"name": "Onyeka Okongwu", "bookmaker": "Bovada", "line": 15.5, "prediction": 16.75, "side": "Over", "odds": 200, "recommendation": 0, "ev": 7.08, "roi": 70.8, "kelly": 0.354, "sigma": "High"},
    {"name": "Naji Marshall", "bookmaker": "Bovada", "line": 14.5, "prediction": 15.1, "side": "Over", "odds": 200, "recommendation": 0, "ev": 6.24, "roi": 62.4, "kelly": 0.312, "sigma": "High"},
    {"name": "Nikola Vucevic", "bookmaker": "Bovada", "line": 15.5, "prediction": 13.98, "side": "Under", "odds": 185, "recommendation": 0, "ev": 6.19, "roi": 61.9, "kelly": 0.334, "sigma": "High"},
    {"name": "Dillon Brooks", "bookmaker": "Bovada", "line": 18.5, "prediction": 22.52, "side": "Over", "odds": 130, "recommendation": 0, "ev": 6.14, "roi": 61.4, "kelly": 0.472, "sigma": "High"},
    {"name": "Isaac Okoro", "bookmaker": "Bovada", "line": 9.5, "prediction": 9.49, "side": "Over", "odds": 200, "recommendation": 0, "ev": 6.07, "roi": 60.7, "kelly": 0.304, "sigma": "High"},
    {"name": "Onyeka Okongwu", "bookmaker": "Bovada", "line": 14.5, "prediction": 16.75, "side": "Over", "odds": 155, "recommendation": 0, "ev": 5.99, "roi": 59.9, "kelly": 0.386, "sigma": "High"},
    {"name": "Anthony Black", "bookmaker": "Bovada", "line": 12.5, "prediction": 13.46, "side": "Over", "odds": 175, "recommendation": 0, "ev": 5.61, "roi": 56.1, "kelly": 0.321, "sigma": "High"},
    {"name": "Brice Sensabaugh", "bookmaker": "Bovada", "line": 10.5, "prediction": 10.24, "side": "Over", "odds": 205, "recommendation": 0, "ev": 5.54, "roi": 55.4, "kelly": 0.27, "sigma": "High"},
    {"name": "Day'Ron Sharpe", "bookmaker": "Bovada", "line": 8.5, "prediction": 8.7, "side": "Over", "odds": 180, "recommendation": 0, "ev": 5.51, "roi": 55.1, "kelly": 0.306, "sigma": "Med"},
    {"name": "Harrison Barnes", "bookmaker": "Bovada", "line": 12.5, "prediction": 12.54, "side": "Over", "odds": 200, "recommendation": 0, "ev": 5.49, "roi": 54.9, "kelly": 0.274, "sigma": "High"},
    {"name": "Draymond Green", "bookmaker": "Bovada", "line": 10.5, "prediction": 10.52, "side": "Over", "odds": 190, "recommendation": 0, "ev": 5.31, "roi": 53.1, "kelly": 0.279, "sigma": "High"},
    {"name": "Naji Marshall", "bookmaker": "Bovada", "line": 13.5, "prediction": 15.1, "side": "Over", "odds": 155, "recommendation": 0, "ev": 5.29, "roi": 52.9, "kelly": 0.341, "sigma": "High"},
    {"name": "Dillon Brooks", "bookmaker": "Bovada", "line": 17.5, "prediction": 22.52, "side": "Over", "odds": 105, "recommendation": 1, "ev": 5.24, "roi": 52.4, "kelly": 0.499, "sigma": "High"},
];const prizepicksPairsData = [
    {"name1": "Dillon Brooks", "name2": "Onyeka Okongwu", "line1": 16.5, "line2": 12.5, "prediction1": 22.52, "prediction2": 16.75, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.33, "kelly": 0.266, "sigma1": "High", "sigma2": "High", "hitRate1": 79.0, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 71.3, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Nicolas Batum", "name2": "Dillon Brooks", "line1": 4.5, "line2": 16.5, "prediction1": 6.26, "prediction2": 22.52, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.2, "kelly": 0.26, "sigma1": "Med", "sigma2": "High", "hitRate1": 31.3, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 79.0, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Dillon Brooks", "name2": "Lauri Markkanen", "line1": 16.5, "line2": 26.5, "prediction1": 22.52, "prediction2": 31.18, "side1": "over", "side2": "over", "recommendation": 1, "ev": 5.05, "kelly": 0.252, "sigma1": "High", "sigma2": "High", "hitRate1": 79.0, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 44.8, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Onyeka Okongwu", "name2": "Lauri Markkanen", "line1": 12.5, "line2": 26.5, "prediction1": 16.75, "prediction2": 31.18, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.26, "kelly": 0.213, "sigma1": "High", "sigma2": "High", "hitRate1": 71.3, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 44.8, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Naji Marshall", "name2": "Onyeka Okongwu", "line1": 11.5, "line2": 12.5, "prediction1": 15.1, "prediction2": 16.75, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.08, "kelly": 0.204, "sigma1": "High", "sigma2": "High", "hitRate1": 80.5, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 71.3, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Nicolas Batum", "name2": "Naji Marshall", "line1": 4.5, "line2": 11.5, "prediction1": 6.26, "prediction2": 15.1, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.88, "kelly": 0.194, "sigma1": "Med", "sigma2": "High", "hitRate1": 31.3, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 80.5, "l5_2": 0.8, "l15_2": 0.33},
    {"name1": "Nicolas Batum", "name2": "Harrison Barnes", "line1": 4.5, "line2": 9.5, "prediction1": 6.26, "prediction2": 12.54, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.88, "kelly": 0.194, "sigma1": "Med", "sigma2": "High", "hitRate1": 31.3, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 80.0, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Lauri Markkanen", "name2": "Josh Giddey", "line1": 26.5, "line2": 20.5, "prediction1": 31.18, "prediction2": 24.77, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.69, "kelly": 0.185, "sigma1": "High", "sigma2": "High", "hitRate1": 44.8, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 55.5, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "Naji Marshall", "name2": "Keyonte George", "line1": 11.5, "line2": 20.5, "prediction1": 15.1, "prediction2": 24.51, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.5, "kelly": 0.175, "sigma1": "High", "sigma2": "High", "hitRate1": 80.5, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 60.6, "l5_2": 0.2, "l15_2": 0.33},
    {"name1": "Zion Williamson", "name2": "Josh Giddey", "line1": 19.0, "line2": 20.5, "prediction1": 22.27, "prediction2": 24.77, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.44, "kelly": 0.172, "sigma1": "High", "sigma2": "High", "hitRate1": 73.7, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 55.5, "l5_2": 0.6, "l15_2": 0.27},
];const prizepicksTriosData = [
    {"name1": "Nicolas Batum", "name2": "Dillon Brooks", "name3": "Onyeka Okongwu", "line1": 4.5, "line2": 16.5, "line3": 12.5, "prediction1": 6.26, "prediction2": 22.52, "prediction3": 16.75, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.62, "kelly": 0.192, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 31.3, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 79.0, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 71.3, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Harrison Barnes", "name2": "Dillon Brooks", "name3": "Onyeka Okongwu", "line1": 9.5, "line2": 16.5, "line3": 12.5, "prediction1": 12.54, "prediction2": 22.52, "prediction3": 16.75, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.62, "kelly": 0.192, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.0, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 79.0, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 71.3, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Nicolas Batum", "name2": "Lauri Markkanen", "name3": "Josh Giddey", "line1": 4.5, "line2": 26.5, "line3": 20.5, "prediction1": 6.26, "prediction2": 31.18, "prediction3": 24.77, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.34, "kelly": 0.167, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 31.3, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 44.8, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 55.5, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Naji Marshall", "name2": "Lauri Markkanen", "name3": "Josh Giddey", "line1": 11.5, "line2": 26.5, "line3": 20.5, "prediction1": 15.1, "prediction2": 31.18, "prediction3": 24.77, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.73, "kelly": 0.155, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.5, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 44.8, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 55.5, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Day'Ron Sharpe", "name2": "Zion Williamson", "name3": "Naji Marshall", "line1": 6.5, "line2": 19.0, "line3": 11.5, "prediction1": 8.7, "prediction2": 22.27, "prediction3": 15.1, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.1, "kelly": 0.142, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 21.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 73.7, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 80.5, "l5_3": 0.8, "l15_3": 0.33},
    {"name1": "Day'Ron Sharpe", "name2": "Zion Williamson", "name3": "Keyonte George", "line1": 6.5, "line2": 19.0, "line3": 20.5, "prediction1": 8.7, "prediction2": 22.27, "prediction3": 24.51, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.8, "kelly": 0.136, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 21.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 73.7, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 60.6, "l5_3": 0.2, "l15_3": 0.33},
    {"name1": "Harrison Barnes", "name2": "Keyonte George", "name3": "Svi Mykhailiuk", "line1": 9.5, "line2": 20.5, "line3": 8.5, "prediction1": 12.54, "prediction2": 24.51, "prediction3": 11.32, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.57, "kelly": 0.131, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.0, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 60.6, "l5_2": 0.2, "l15_2": 0.33, "hitRate3": 75.3, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Draymond Green", "name2": "Collin Gillespie", "name3": "Svi Mykhailiuk", "line1": 8.5, "line2": 11.5, "line3": 8.5, "prediction1": 10.52, "prediction2": 8.1, "prediction3": 11.32, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 5.48, "kelly": 0.11, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 22.3, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 84.7, "l5_2": 0.2, "l15_2": 0.4, "hitRate3": 75.3, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Precious Achiuwa", "name2": "Max Christie", "name3": "Collin Gillespie", "line1": 5.5, "line2": 11.5, "line3": 11.5, "prediction1": 6.91, "prediction2": 14.24, "prediction3": 8.1, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 5.07, "kelly": 0.101, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 63.9, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 69.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 84.7, "l5_3": 0.2, "l15_3": 0.4},
    {"name1": "Draymond Green", "name2": "Max Christie", "name3": "Nikola Vu\u010devi\u0107", "line1": 8.5, "line2": 11.5, "line3": 17.5, "prediction1": 10.52, "prediction2": 14.24, "prediction3": 13.98, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 4.51, "kelly": 0.09, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 22.3, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 69.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Dillon Brooks", "name2": "Onyeka Okongwu", "line1": 17.5, "line2": 12.5, "prediction1": 22.52, "prediction2": 16.75, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.56, "kelly": 0.228, "sigma1": "High", "sigma2": "High", "hitRate1": 71.6, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 71.3, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Zion Williamson", "name2": "Dillon Brooks", "line1": 18.5, "line2": 17.5, "prediction1": 22.27, "prediction2": 22.52, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.54, "kelly": 0.227, "sigma1": "High", "sigma2": "High", "hitRate1": 80.4, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 71.6, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Dillon Brooks", "name2": "Lauri Markkanen", "line1": 17.5, "line2": 26.5, "prediction1": 22.52, "prediction2": 31.18, "side1": "over", "side2": "over", "recommendation": 1, "ev": 4.46, "kelly": 0.223, "sigma1": "High", "sigma2": "High", "hitRate1": 71.6, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 44.8, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Onyeka Okongwu", "name2": "Lauri Markkanen", "line1": 12.5, "line2": 26.5, "prediction1": 16.75, "prediction2": 31.18, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.29, "kelly": 0.214, "sigma1": "High", "sigma2": "High", "hitRate1": 71.3, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 44.8, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Zion Williamson", "name2": "Onyeka Okongwu", "line1": 18.5, "line2": 12.5, "prediction1": 22.27, "prediction2": 16.75, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.28, "kelly": 0.214, "sigma1": "High", "sigma2": "High", "hitRate1": 80.4, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 71.3, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Zion Williamson", "name2": "Lauri Markkanen", "line1": 18.5, "line2": 26.5, "prediction1": 22.27, "prediction2": 31.18, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.16, "kelly": 0.208, "sigma1": "High", "sigma2": "High", "hitRate1": 80.4, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 44.8, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Harrison Barnes", "name2": "Josh Giddey", "line1": 9.5, "line2": 20.5, "prediction1": 12.54, "prediction2": 24.77, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.56, "kelly": 0.178, "sigma1": "High", "sigma2": "High", "hitRate1": 80.0, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 55.5, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "Keyonte George", "name2": "Josh Giddey", "line1": 20.5, "line2": 20.5, "prediction1": 24.51, "prediction2": 24.77, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.52, "kelly": 0.176, "sigma1": "High", "sigma2": "High", "hitRate1": 60.6, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 55.5, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "Naji Marshall", "name2": "Josh Giddey", "line1": 11.5, "line2": 20.5, "prediction1": 15.1, "prediction2": 24.77, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.36, "kelly": 0.168, "sigma1": "High", "sigma2": "High", "hitRate1": 80.5, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 55.5, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "Day'Ron Sharpe", "name2": "Naji Marshall", "line1": 6.5, "line2": 11.5, "prediction1": 8.7, "prediction2": 15.1, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.34, "kelly": 0.167, "sigma1": "Med", "sigma2": "High", "hitRate1": 21.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 80.5, "l5_2": 0.8, "l15_2": 0.33},
];const underdogTriosData = [
    {"name1": "Zion Williamson", "name2": "Dillon Brooks", "name3": "Onyeka Okongwu", "line1": 18.5, "line2": 17.5, "line3": 12.5, "prediction1": 22.27, "prediction2": 22.52, "prediction3": 16.75, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.17, "kelly": 0.183, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.4, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 71.6, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 71.3, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Zion Williamson", "name2": "Dillon Brooks", "name3": "Lauri Markkanen", "line1": 18.5, "line2": 17.5, "line3": 26.5, "prediction1": 22.27, "prediction2": 22.52, "prediction3": 31.18, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.93, "kelly": 0.179, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.4, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 71.6, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 44.8, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Onyeka Okongwu", "name2": "Lauri Markkanen", "name3": "Josh Giddey", "line1": 12.5, "line2": 26.5, "line3": 20.5, "prediction1": 16.75, "prediction2": 31.18, "prediction3": 24.77, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.29, "kelly": 0.166, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 71.3, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 44.8, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 55.5, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Harrison Barnes", "name2": "Josh Giddey", "name3": "Kevin Huerter", "line1": 9.5, "line2": 20.5, "line3": 12.5, "prediction1": 12.54, "prediction2": 24.77, "prediction3": 15.89, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.13, "kelly": 0.143, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.0, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 55.5, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 71.7, "l5_3": 0.8, "l15_3": 0.47},
    {"name1": "Naji Marshall", "name2": "Kevin Huerter", "name3": "Svi Mykhailiuk", "line1": 11.5, "line2": 12.5, "line3": 8.5, "prediction1": 15.1, "prediction2": 15.89, "prediction3": 11.32, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.96, "kelly": 0.139, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.5, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 71.7, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 75.3, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Harrison Barnes", "name2": "Naji Marshall", "name3": "Collin Gillespie", "line1": 9.5, "line2": 11.5, "line3": 11.5, "prediction1": 12.54, "prediction2": 15.1, "prediction3": 8.1, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 6.93, "kelly": 0.139, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.0, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 80.5, "l5_2": 0.8, "l15_2": 0.33, "hitRate3": 84.7, "l5_3": 0.2, "l15_3": 0.4},
    {"name1": "Collin Gillespie", "name2": "Keyonte George", "name3": "Svi Mykhailiuk", "line1": 11.5, "line2": 20.5, "line3": 8.5, "prediction1": 8.1, "prediction2": 24.51, "prediction3": 11.32, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.68, "kelly": 0.134, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 84.7, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 60.6, "l5_2": 0.2, "l15_2": 0.33, "hitRate3": 75.3, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Day'Ron Sharpe", "name2": "Max Christie", "name3": "Keyonte George", "line1": 6.5, "line2": 11.5, "line3": 20.5, "prediction1": 8.7, "prediction2": 14.24, "prediction3": 24.51, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.81, "kelly": 0.116, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 21.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 69.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 60.6, "l5_3": 0.2, "l15_3": 0.33},
    {"name1": "Day'Ron Sharpe", "name2": "Kyshawn George", "name3": "Nikola Vu\u010devi\u0107", "line1": 6.5, "line2": 13.5, "line3": 17.5, "prediction1": 8.7, "prediction2": 16.4, "prediction3": 13.98, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 5.33, "kelly": 0.107, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 21.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 45.4, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Draymond Green", "name2": "Will Richard", "name3": "Nikola Vu\u010devi\u0107", "line1": 8.5, "line2": 8.5, "line3": 17.5, "prediction1": 10.52, "prediction2": 10.59, "prediction3": 13.98, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 4.58, "kelly": 0.092, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 22.3, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 82.6, "l5_2": 0.8, "l15_2": 0.33, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
];// This is a large data file - I'll create a simplified version that includes all the hit rates data
// For brevity, I'll include a condensed version with the key structures
const prizepicksPointsHitRates = [
    {"name": "Luke Kennard", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.817, "underPct": 0.183},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.805, "underPct": 0.195},
    {"name": "Harrison Barnes", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.8, "underPct": 0.2},
    {"name": "Julian Champagnie", "line": 8.0, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.79, "underPct": 0.21},
    {"name": "Dillon Brooks", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.79, "underPct": 0.21},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.758, "underPct": 0.242},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.753, "underPct": 0.247},
    {"name": "Ayo Dosunmu", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.748, "underPct": 0.252},
    {"name": "Zion Williamson", "line": 19.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.737, "underPct": 0.263},
    {"name": "Will Richard", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.733, "underPct": 0.267},
    {"name": "Onyeka Okongwu", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.713, "underPct": 0.287},
    {"name": "Reed Sheppard", "line": 11.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.704, "underPct": 0.296},
    {"name": "Max Christie", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.697, "underPct": 0.303},
    {"name": "Domantas Sabonis", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.693, "underPct": 0.307},
    {"name": "Tre Jones", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.681, "underPct": 0.319},
    {"name": "Steven Adams", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.646, "underPct": 0.354},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.639, "underPct": 0.361},
    {"name": "Corey Kispert", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.639, "underPct": 0.361},
    {"name": "Neemias Queta", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.63, "underPct": 0.37},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.62, "underPct": 0.38},
    {"name": "James Harden", "line": 24.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.615, "underPct": 0.385},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.606, "underPct": 0.394},
    {"name": "Keldon Johnson", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.601, "underPct": 0.399},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.591, "underPct": 0.409},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.59, "underPct": 0.41},
    {"name": "Amen Thompson", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.585, "underPct": 0.415},
    {"name": "Nickeil Alexander-Walker", "line": 16.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.572, "underPct": 0.428},
    {"name": "Russell Westbrook", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.571, "underPct": 0.429},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.555, "underPct": 0.445},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.546, "underPct": 0.454},
    {"name": "Stephen Curry", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.541, "underPct": 0.459},
    {"name": "Zaccharie Risacher", "line": 13.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.53, "underPct": 0.47},
    {"name": "Jalen Johnson", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.523, "underPct": 0.477},
    {"name": "Alex Sarr", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.515, "underPct": 0.485},
    {"name": "Brandin Podziemski", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.514, "underPct": 0.486},
    {"name": "Payton Pritchard", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.497, "underPct": 0.503},
    {"name": "Luka Garza", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Alperen Sengun", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.488, "underPct": 0.512},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.474, "underPct": 0.526},
    {"name": "Buddy Hield", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.474, "underPct": 0.526},
    {"name": "Kevin Durant", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.471, "underPct": 0.529},
    {"name": "Ryan Dunn", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.466, "underPct": 0.534},
    {"name": "Kris Dunn", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.466, "underPct": 0.534},
    {"name": "Bilal Coulibaly", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.463, "underPct": 0.537},
    {"name": "Kyshawn George", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.454, "underPct": 0.546},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.448, "underPct": 0.552},
    {"name": "Klay Thompson", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.442, "underPct": 0.558},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.438, "underPct": 0.562},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.429, "underPct": 0.571},
    {"name": "Josh Okogie", "line": 8.0, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.429, "underPct": 0.571},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.423, "underPct": 0.577},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.406, "underPct": 0.594},
    {"name": "Noah Clowney", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.393, "underPct": 0.607},
    {"name": "Matas Buzelis", "line": 14.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.392, "underPct": 0.608},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.382, "underPct": 0.618},
    {"name": "Tre Johnson", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.372, "underPct": 0.628},
    {"name": "Anfernee Simons", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.363, "underPct": 0.637},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.347, "underPct": 0.653},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.34, "underPct": 0.66},
    {"name": "Victor Wembanyama", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.318, "underPct": 0.682},
    {"name": "Nicolas Batum", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.313, "underPct": 0.687},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.312, "underPct": 0.688},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.312, "underPct": 0.688},
    {"name": "Stephon Castle", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.307, "underPct": 0.693},
    {"name": "De'Aaron Fox", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.301, "underPct": 0.699},
    {"name": "Mark Williams", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.299, "underPct": 0.701},
    {"name": "Tristan da Silva", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.287, "underPct": 0.713},
    {"name": "John Collins", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.275, "underPct": 0.725},
    {"name": "Terance Mann", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.266, "underPct": 0.734},
    {"name": "Derrick White", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.258, "underPct": 0.742},
    {"name": "Franz Wagner", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.254, "underPct": 0.746},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.237, "underPct": 0.763},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.223, "underPct": 0.777},
    {"name": "D'Angelo Russell", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.22, "underPct": 0.78},
    {"name": "Devin Vassell", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.215, "underPct": 0.785},
    {"name": "Day'Ron Sharpe", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.214, "underPct": 0.786},
    {"name": "Desmond Bane", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.173, "underPct": 0.827},
    {"name": "Dyson Daniels", "line": 12.0, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.163, "underPct": 0.837},
    {"name": "Drake Powell", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.163, "underPct": 0.837},
    {"name": "Al Horford", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.16, "underPct": 0.84},
    {"name": "Collin Gillespie", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.153, "underPct": 0.847},
    {"name": "Anthony Black", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.149, "underPct": 0.851},
    {"name": "Sam Hauser", "line": 6.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.127, "underPct": 0.873},
    {"name": "Tyrese Martin", "line": 8.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.121, "underPct": 0.879},
];const prizepicksAssistsHitRates = [
    {"name": "Dyson Daniels", "line": 5.0, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.657, "underPct": 0.343},
    {"name": "Isaiah Collier", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.609, "underPct": 0.391},
    {"name": "Alperen Sengun", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.579, "underPct": 0.421},
    {"name": "Kevin Huerter", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.557, "underPct": 0.443},
    {"name": "Naji Marshall", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.557, "underPct": 0.443},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.549, "underPct": 0.451},
    {"name": "Neemias Queta", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.537, "underPct": 0.463},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.512, "underPct": 0.488},
    {"name": "Devin Booker", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.501, "underPct": 0.499},
    {"name": "Josh Giddey", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.494, "underPct": 0.506},
    {"name": "De'Aaron Fox", "line": 6.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.493, "underPct": 0.507},
    {"name": "Terance Mann", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.483, "underPct": 0.517},
    {"name": "Zion Williamson", "line": 4.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.464, "underPct": 0.536},
    {"name": "Victor Wembanyama", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.463, "underPct": 0.537},
    {"name": "Jalen Johnson", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.452, "underPct": 0.548},
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Kyshawn George", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.406, "underPct": 0.594},
    {"name": "Trey Murphy III", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.391, "underPct": 0.609},
    {"name": "Domantas Sabonis", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.391, "underPct": 0.609},
    {"name": "Jaylen Brown", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.39, "underPct": 0.61},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.385, "underPct": 0.615},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.368, "underPct": 0.632},
    {"name": "Will Richard", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.367, "underPct": 0.633},
    {"name": "Brandon Williams", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.362, "underPct": 0.638},
    {"name": "Keyonte George", "line": 6.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.347, "underPct": 0.653},
    {"name": "Draymond Green", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.34, "underPct": 0.66},
    {"name": "Drake Powell", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.312, "underPct": 0.688},
    {"name": "Payton Pritchard", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.303, "underPct": 0.697},
    {"name": "Tre Jones", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.296, "underPct": 0.704},
    {"name": "Collin Gillespie", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.291, "underPct": 0.709},
    {"name": "Tre Johnson", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.277, "underPct": 0.723},
    {"name": "Stephen Curry", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.277, "underPct": 0.723},
];const prizepicksReboundsHitRates = [
    {"name": "Alperen Sengun", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.707, "underPct": 0.293},
    {"name": "Zion Williamson", "line": 5.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.689, "underPct": 0.311},
    {"name": "Franz Wagner", "line": 5.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.687, "underPct": 0.313},
    {"name": "Josh Giddey", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.686, "underPct": 0.314},
    {"name": "Trey Murphy III", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.666, "underPct": 0.334},
    {"name": "Max Christie", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.655, "underPct": 0.345},
    {"name": "Stephon Castle", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Steven Adams", "line": 7.0, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.635, "underPct": 0.365},
    {"name": "Jalen Johnson", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.624, "underPct": 0.376},
    {"name": "Julian Champagnie", "line": 3.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.623, "underPct": 0.377},
    {"name": "Dereck Lively II", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.621, "underPct": 0.379},
    {"name": "Bilal Coulibaly", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.613, "underPct": 0.387},
    {"name": "P.J. Washington", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.56, "underPct": 0.44},
    {"name": "James Harden", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.557, "underPct": 0.443},
    {"name": "Svi Mykhailiuk", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.551, "underPct": 0.449},
    {"name": "Victor Wembanyama", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.503, "underPct": 0.497},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.502, "underPct": 0.498},
    {"name": "Keldon Johnson", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Precious Achiuwa", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.495, "underPct": 0.505},
    {"name": "Cooper Flagg", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.493, "underPct": 0.507},
    {"name": "Russell Westbrook", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.487, "underPct": 0.513},
    {"name": "Tre Johnson", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.484, "underPct": 0.516},
    {"name": "Onyeka Okongwu", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.472, "underPct": 0.528},
    {"name": "Ivica Zubac", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.471, "underPct": 0.529},
    {"name": "Draymond Green", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.465, "underPct": 0.535},
    {"name": "Ayo Dosunmu", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.444, "underPct": 0.556},
    {"name": "Royce O'Neale", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.443, "underPct": 0.557},
    {"name": "Kris Murray", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.434, "underPct": 0.566},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.434, "underPct": 0.566},
    {"name": "Devin Booker", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.42, "underPct": 0.58},
    {"name": "Collin Gillespie", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.415, "underPct": 0.585},
    {"name": "Keyonte George", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.415, "underPct": 0.585},
    {"name": "Al Horford", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.41, "underPct": 0.59},
    {"name": "Neemias Queta", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.396, "underPct": 0.604},
    {"name": "Kevin Durant", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.387, "underPct": 0.613},
    {"name": "Amen Thompson", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.387, "underPct": 0.613},
    {"name": "Kyle Filipowski", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.384, "underPct": 0.616},
    {"name": "Jordan Walsh", "line": 5.0, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.38, "underPct": 0.62},
    {"name": "Jaylen Brown", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.376, "underPct": 0.624},
    {"name": "Desmond Bane", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.365, "underPct": 0.635},
    {"name": "Deni Avdija", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.358, "underPct": 0.642},
    {"name": "Lauri Markkanen", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.349, "underPct": 0.651},
    {"name": "Kevon Looney", "line": 4.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.347, "underPct": 0.653},
    {"name": "Mark Williams", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.344, "underPct": 0.656},
    {"name": "Domantas Sabonis", "line": 12.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.328, "underPct": 0.672},
    {"name": "Derrick White", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.32, "underPct": 0.68},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.269, "underPct": 0.731},
    {"name": "Khris Middleton", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.262, "underPct": 0.738},
    {"name": "Daniel Gafford", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.261, "underPct": 0.739},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.258, "underPct": 0.742},
    {"name": "John Collins", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.257, "underPct": 0.743},
    {"name": "Corey Kispert", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.251, "underPct": 0.749},
    {"name": "Stephen Curry", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.23, "underPct": 0.77},
    {"name": "Ryan Dunn", "line": 5.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.228, "underPct": 0.772},
    {"name": "Ace Bailey", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.221, "underPct": 0.779},
    {"name": "Kyshawn George", "line": 6.0, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.163, "underPct": 0.837},
];const prizepicksBlocksHitRates = [
    {"name": "John Collins", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.389, "underPct": 0.611},
    {"name": "Brook Lopez", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.56, "underPct": 0.44},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.394, "underPct": 0.606},
    {"name": "Day'Ron Sharpe", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.333, "underPct": 0.667},
    {"name": "Kyshawn George", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.609, "underPct": 0.391},
    {"name": "Moses Moody", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.542, "underPct": 0.458},
    {"name": "Kevin Durant", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.499, "underPct": 0.501},
    {"name": "Steven Adams", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.542, "underPct": 0.458},
    {"name": "Cooper Flagg", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.504, "underPct": 0.496},
    {"name": "Onyeka Okongwu", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.589, "underPct": 0.411},
];const prizepicksStealsHitRates = [
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.227, "underPct": 0.773},
    {"name": "Neemias Queta", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.564, "underPct": 0.436},
    {"name": "Nicolas Batum", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.574, "underPct": 0.426},
    {"name": "Sam Hauser", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.448, "underPct": 0.552},
    {"name": "Tre Johnson", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.264, "underPct": 0.736},
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.411, "underPct": 0.589},
    {"name": "Tyus Jones", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.536, "underPct": 0.464},
    {"name": "Brandon Williams", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.569, "underPct": 0.431},
    {"name": "Dereck Lively II", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.479, "underPct": 0.521},
    {"name": "Daniel Gafford", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.578, "underPct": 0.422},
    {"name": "Isaiah Collier", "line": 0.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.317, "underPct": 0.683},
    {"name": "Svi Mykhailiuk", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.452, "underPct": 0.548},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Luke Kennard", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 38.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 29.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 29.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Smith", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Okogie", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 26.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Steven Adams", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Patrick Williams", "line": 10.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Naji Marshall", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevon Looney", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Domantas Sabonis", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luka Garza", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Filipowski", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 38.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tre Jones", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keldon Johnson", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dereck Lively II", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Johnson", "line": 37.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 24.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 38.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mark Williams", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 30.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 34.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "James Harden", "line": 38.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Al Horford", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Corey Kispert", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Harrison Barnes", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Johnson", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Day'Ron Sharpe", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Martin", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 28.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Will Richard", "line": 14.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Stephon Castle", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Victor Wembanyama", "line": 43.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brook Lopez", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alex Sarr", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 34.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "John Collins", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Amen Thompson", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 40.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drake Powell", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "D'Angelo Russell", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Khris Middleton", "line": 15.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyus Jones", "line": 9.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Kevin Huerter", "line": 15.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Max Christie", "line": 14.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jordan Walsh", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Luke Kennard", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Okogie", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Steven Adams", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Reed Sheppard", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 23.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevon Looney", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 19.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brice Sensabaugh", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Filipowski", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Domantas Sabonis", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 19.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Sarr", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 33.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 33.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kevin Durant", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Victor Wembanyama", "line": 39.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 25.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Precious Achiuwa", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ziaire Williams", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Johnson", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Al Horford", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Corey Kispert", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 12.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Buddy Hield", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tristan da Silva", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Stephon Castle", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ryan Dunn", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drake Powell", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Collin Gillespie", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dereck Lively II", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "D'Angelo Russell", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandin Podziemski", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 18.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyus Jones", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Sam Hauser", "line": 10.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Khris Middleton", "line": 13.0, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const prizepicksPAHitRates = [
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tre Jones", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kevin Huerter", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 23.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 30.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Reed Sheppard", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 19.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zaccharie Risacher", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaac Okoro", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Okogie", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kennard", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brook Lopez", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Domantas Sabonis", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephon Castle", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Garza", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Filipowski", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 20.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Johnson", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Victor Wembanyama", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bilal Coulibaly", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Corey Kispert", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Draymond Green", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 27.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jonathan Isaac", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Collin Gillespie", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 11.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drake Powell", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 36.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Daniel Gafford", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Khris Middleton", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sam Hauser", "line": 7.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const prizepicksRAHitRates = [
    {"name": "Franz Wagner", "line": 10.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephon Castle", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 16.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Steven Adams", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Trey Murphy III", "line": 9.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 9.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevon Looney", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 15.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keyonte George", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 16.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dereck Lively II", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tristan da Silva", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luke Kennard", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 15.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 8.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Al Horford", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Payton Pritchard", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Victor Wembanyama", "line": 17.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Martin", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Drake Powell", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Corey Kispert", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 7.0, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kyshawn George", "line": 10.0, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Domantas Sabonis", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Stephen Curry", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Khris Middleton", "line": 6.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 7.0, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const prizepicksTurnoversHitRates = [
    {"name": "Draymond Green", "line": 2.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Toumani Camara", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Onyeka Okongwu", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 2.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Buddy Hield", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alperen Sengun", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Domantas Sabonis", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Luka Garza", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Derrick White", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Sarr", "line": 2.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Reed Sheppard", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephon Castle", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jrue Holiday", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Day'Ron Sharpe", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Amen Thompson", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kevin Durant", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Victor Wembanyama", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const underdogPointsHitRates = [
    {"name": "Will Richard", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.826, "underPct": 0.174},
    {"name": "Luke Kennard", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.817, "underPct": 0.183},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.805, "underPct": 0.195},
    {"name": "Zion Williamson", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.804, "underPct": 0.196},
    {"name": "Harrison Barnes", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.8, "underPct": 0.2},
    {"name": "Jalen Smith", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.778, "underPct": 0.222},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.758, "underPct": 0.242},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.753, "underPct": 0.247},
    {"name": "Ayo Dosunmu", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.748, "underPct": 0.252},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.717, "underPct": 0.283},
    {"name": "Dillon Brooks", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.716, "underPct": 0.284},
    {"name": "Onyeka Okongwu", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.713, "underPct": 0.287},
    {"name": "Reed Sheppard", "line": 11.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.704, "underPct": 0.296},
    {"name": "Max Christie", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.697, "underPct": 0.303},
    {"name": "James Harden", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.689, "underPct": 0.311},
    {"name": "Tre Jones", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.681, "underPct": 0.319},
    {"name": "Steven Adams", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.646, "underPct": 0.354},
    {"name": "Corey Kispert", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.639, "underPct": 0.361},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.639, "underPct": 0.361},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.62, "underPct": 0.38},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.606, "underPct": 0.394},
    {"name": "Keldon Johnson", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.601, "underPct": 0.399},
    {"name": "Amen Thompson", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.585, "underPct": 0.415},
    {"name": "Nickeil Alexander-Walker", "line": 16.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.572, "underPct": 0.428},
    {"name": "Russell Westbrook", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.571, "underPct": 0.429},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.555, "underPct": 0.445},
    {"name": "Shaedon Sharpe", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.553, "underPct": 0.447},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.553, "underPct": 0.447},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.546, "underPct": 0.454},
    {"name": "Stephen Curry", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.541, "underPct": 0.459},
    {"name": "Zaccharie Risacher", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.53, "underPct": 0.47},
    {"name": "Jalen Johnson", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.523, "underPct": 0.477},
    {"name": "Alex Sarr", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.515, "underPct": 0.485},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.514, "underPct": 0.486},
    {"name": "Brandin Podziemski", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.514, "underPct": 0.486},
    {"name": "Matas Buzelis", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.498, "underPct": 0.502},
    {"name": "Payton Pritchard", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.497, "underPct": 0.503},
    {"name": "Luka Garza", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Alperen Sengun", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.488, "underPct": 0.512},
    {"name": "Buddy Hield", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.474, "underPct": 0.526},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.474, "underPct": 0.526},
    {"name": "Kris Dunn", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.466, "underPct": 0.534},
    {"name": "Ryan Dunn", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.466, "underPct": 0.534},
    {"name": "Kyshawn George", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.454, "underPct": 0.546},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.448, "underPct": 0.552},
    {"name": "Klay Thompson", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.442, "underPct": 0.558},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.438, "underPct": 0.562},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.433, "underPct": 0.567},
    {"name": "Josh Okogie", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.429, "underPct": 0.571},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.406, "underPct": 0.594},
    {"name": "Kris Murray", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.398, "underPct": 0.602},
    {"name": "Noah Clowney", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.393, "underPct": 0.607},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.382, "underPct": 0.618},
    {"name": "Tre Johnson", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.372, "underPct": 0.628},
    {"name": "Anfernee Simons", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.363, "underPct": 0.637},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.347, "underPct": 0.653},
    {"name": "Jerami Grant", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.334, "underPct": 0.666},
    {"name": "Victor Wembanyama", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.318, "underPct": 0.682},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.312, "underPct": 0.688},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.312, "underPct": 0.688},
    {"name": "Stephon Castle", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.307, "underPct": 0.693},
    {"name": "Mark Williams", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.299, "underPct": 0.701},
    {"name": "John Collins", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.275, "underPct": 0.725},
    {"name": "Terance Mann", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.266, "underPct": 0.734},
    {"name": "Derrick White", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.258, "underPct": 0.742},
    {"name": "Franz Wagner", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.254, "underPct": 0.746},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.248, "underPct": 0.752},
    {"name": "Donovan Clingan", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.246, "underPct": 0.754},
    {"name": "De'Aaron Fox", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.232, "underPct": 0.768},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.223, "underPct": 0.777},
    {"name": "Devin Vassell", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.215, "underPct": 0.785},
    {"name": "Day'Ron Sharpe", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.214, "underPct": 0.786},
    {"name": "Tristan da Silva", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.198, "underPct": 0.802},
    {"name": "Desmond Bane", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.173, "underPct": 0.827},
    {"name": "Al Horford", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.16, "underPct": 0.84},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.16, "underPct": 0.84},
    {"name": "Collin Gillespie", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.153, "underPct": 0.847},
    {"name": "Anthony Black", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.149, "underPct": 0.851},
    {"name": "Sam Hauser", "line": 6.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.127, "underPct": 0.873},
];const underdogAssistsHitRates = [
    {"name": "Alperen Sengun", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.579, "underPct": 0.421},
    {"name": "Isaac Okoro", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.537, "underPct": 0.463},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.512, "underPct": 0.488},
    {"name": "Terance Mann", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.483, "underPct": 0.517},
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Trey Murphy III", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.391, "underPct": 0.609},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.368, "underPct": 0.632},
    {"name": "Will Richard", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.367, "underPct": 0.633},
    {"name": "Kris Murray", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.356, "underPct": 0.644},
    {"name": "Deni Avdija", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.325, "underPct": 0.675},
    {"name": "Drake Powell", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.312, "underPct": 0.688},
    {"name": "Tre Johnson", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.277, "underPct": 0.723},
    {"name": "Tristan da Silva", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.213, "underPct": 0.787},
];const underdogReboundsHitRates = [
    {"name": "Alperen Sengun", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.707, "underPct": 0.293},
    {"name": "Matas Buzelis", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.703, "underPct": 0.297},
    {"name": "Zion Williamson", "line": 5.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.689, "underPct": 0.311},
    {"name": "Franz Wagner", "line": 5.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.687, "underPct": 0.313},
    {"name": "Josh Giddey", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.686, "underPct": 0.314},
    {"name": "Jalen Smith", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.672, "underPct": 0.328},
    {"name": "Max Christie", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.655, "underPct": 0.345},
    {"name": "Stephon Castle", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Julian Champagnie", "line": 3.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.623, "underPct": 0.377},
    {"name": "Dereck Lively II", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.621, "underPct": 0.379},
    {"name": "P.J. Washington", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.56, "underPct": 0.44},
    {"name": "Zach LaVine", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.551, "underPct": 0.449},
    {"name": "Victor Wembanyama", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.503, "underPct": 0.497},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.502, "underPct": 0.498},
    {"name": "Tre Johnson", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.484, "underPct": 0.516},
    {"name": "Tyrese Martin", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.411, "underPct": 0.589},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.407, "underPct": 0.593},
    {"name": "Jaylen Brown", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.376, "underPct": 0.624},
    {"name": "Deni Avdija", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.358, "underPct": 0.642},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.289, "underPct": 0.711},
    {"name": "Jonathan Isaac", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.275, "underPct": 0.725},
    {"name": "Corey Kispert", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.251, "underPct": 0.749},
];const underdogBlocksHitRates = [
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.394, "underPct": 0.606},
];const underdogStealsHitRates = [
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Brandon Williams", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.569, "underPct": 0.431},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Kevin Huerter", "line": 18.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Smith", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Luke Kennard", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Steven Adams", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 38.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 29.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 19.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Josh Okogie", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Domantas Sabonis", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 29.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Filipowski", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jrue Holiday", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tre Jones", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Patrick Williams", "line": 10.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Precious Achiuwa", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julian Champagnie", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keldon Johnson", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 38.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 38.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keyonte George", "line": 30.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dereck Lively II", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Johnson", "line": 37.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Lauri Markkanen", "line": 34.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Stephen Curry", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Corey Kispert", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephon Castle", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Johnson", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ziaire Williams", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 28.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Victor Wembanyama", "line": 44.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Al Horford", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brook Lopez", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 36.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "James Harden", "line": 38.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Sarr", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 34.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tristan da Silva", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 40.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "D'Angelo Russell", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Khris Middleton", "line": 15.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyus Jones", "line": 9.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPRHitRates = [
    {"name": "Alperen Sengun", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Domantas Sabonis", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jrue Holiday", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 25.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Victor Wembanyama", "line": 40.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephon Castle", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "John Collins", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const underdogPAHitRates = [
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephon Castle", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jrue Holiday", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 20.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Victor Wembanyama", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 36.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Keyonte George", "line": 26.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const underdogRAHitRates = [
    {"name": "Stephon Castle", "line": 13.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 10.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Steven Adams", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Josh Giddey", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jrue Holiday", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dereck Lively II", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ivica Zubac", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Domantas Sabonis", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const underdogTurnoversHitRates = [
    {"name": "Zion Williamson", "line": 2.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 2.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kyshawn George", "line": 2.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Victor Wembanyama", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Domantas Sabonis", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shaedon Sharpe", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const underdogBlocksStealsHitRates = [
    {"name": "Victor Wembanyama", "line": 4.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Alex Sarr", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Amen Thompson", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Daniel Gafford", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Donovan Clingan", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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
            <th style="width: 10%">Prediction</th>
            <th style="width: 10%">Side</th>
            <th style="width: 8%">Odds</th>
            <th style="width: 9%">EV %</th>
            <th style="width: 9%">ROI %</th>
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
            <td class="kelly-cell">${row.roi.toFixed(1)}%</td>
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
            <th style="width: 16%">Player 1</th>
            <th style="width: 6%">Line 1</th>
            <th style="width: 6%">Pred 1</th>
            <th style="width: 16%">Player 2</th>
            <th style="width: 6%">Line 2</th>
            <th style="width: 6%">Pred 2</th>
            <th style="width: 9%">EV %</th>
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
            <th style="width: 13%">Player 1</th>
            <th style="width: 5%">Line 1</th>
            <th style="width: 5%">Pred 1</th>
            <th style="width: 13%">Player 2</th>
            <th style="width: 5%">Line 2</th>
            <th style="width: 5%">Pred 2</th>
            <th style="width: 13%">Player 3</th>
            <th style="width: 5%">Line 3</th>
            <th style="width: 5%">Pred 3</th>
            <th style="width: 7%">EV %</th>
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
                <div class="stat-label">Prediction & Side</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Model's projected value and betting direction</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">EV % & ROI %</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value & Return on Investment percentage</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Odds</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">American odds format. <span style="color: #34d399;">+</span> = underdog, <span style="color: #f87171;">-</span> = favorite</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sigma</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (consistent), Med, Low (volatile)</div>
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
                <div class="stat-label">Pred (Prediction)</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Model's predicted value. <span style="color: #10b981;">Green</span> = over line, <span style="color: #f59e0b;">Orange</span> = under</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">EV % & Kelly</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value(How much you can expect to win per 10$ bet) & Kelly Criterion bet sizing %</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sigma</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (consistent), Med, Low (volatile)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Rec (Recommendation)</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">1 = Strong play, 0 = Consider</div>
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

// Initial render
render();

