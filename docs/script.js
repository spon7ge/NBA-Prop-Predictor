const prizepicksSinglesData = [
    {"name": "Bobby Portis", "bookmaker": "FanDuel", "line": 14.5, "prediction": 9.42, "side": "Under", "odds": -104, "recommendation": 1, "ev": 5.41, "kelly": 0.562, "sigma": "High"},
    {"name": "Bobby Portis", "bookmaker": "BetRivers", "line": 13.5, "prediction": 9.42, "side": "Under", "odds": 105, "recommendation": 0, "ev": 4.82, "kelly": 0.459, "sigma": "High"},
    {"name": "Bobby Portis", "bookmaker": "DraftKings", "line": 14.5, "prediction": 9.42, "side": "Under", "odds": -112, "recommendation": 1, "ev": 4.73, "kelly": 0.53, "sigma": "High"},
    {"name": "Bobby Portis", "bookmaker": "BetMGM", "line": 13.5, "prediction": 9.42, "side": "Under", "odds": 100, "recommendation": 0, "ev": 4.47, "kelly": 0.447, "sigma": "High"},
    {"name": "Bobby Portis", "bookmaker": "BetRivers", "line": 14.5, "prediction": 9.42, "side": "Under", "odds": -120, "recommendation": 1, "ev": 4.36, "kelly": 0.523, "sigma": "High"},
    {"name": "Santi Aldama", "bookmaker": "DraftKings", "line": 18.5, "prediction": 13.8, "side": "Under", "odds": -118, "recommendation": 1, "ev": 3.82, "kelly": 0.45, "sigma": "High"},
    {"name": "Bobby Portis", "bookmaker": "BetRivers", "line": 15.5, "prediction": 9.42, "side": "Under", "odds": -150, "recommendation": 1, "ev": 3.77, "kelly": 0.566, "sigma": "High"},
    {"name": "Santi Aldama", "bookmaker": "BetMGM", "line": 17.5, "prediction": 13.8, "side": "Under", "odds": -105, "recommendation": 0, "ev": 3.66, "kelly": 0.384, "sigma": "High"},
    {"name": "Santi Aldama", "bookmaker": "FanDuel", "line": 18.5, "prediction": 13.8, "side": "Under", "odds": -122, "recommendation": 1, "ev": 3.57, "kelly": 0.436, "sigma": "High"},
    {"name": "Santi Aldama", "bookmaker": "BetRivers", "line": 16.5, "prediction": 13.8, "side": "Under", "odds": 110, "recommendation": 0, "ev": 3.49, "kelly": 0.317, "sigma": "High"},
    {"name": "Nicolas Batum", "bookmaker": "BetMGM", "line": 4.5, "prediction": 6.35, "side": "Over", "odds": -115, "recommendation": 0, "ev": 3.34, "kelly": 0.384, "sigma": "Med"},
    {"name": "Santi Aldama", "bookmaker": "BetRivers", "line": 17.5, "prediction": 13.8, "side": "Under", "odds": -110, "recommendation": 0, "ev": 3.34, "kelly": 0.368, "sigma": "High"},
    {"name": "Brook Lopez", "bookmaker": "FanDuel", "line": 6.5, "prediction": 7.87, "side": "Over", "odds": 102, "recommendation": 0, "ev": 3.27, "kelly": 0.32, "sigma": "Med"},
    {"name": "Brook Lopez", "bookmaker": "DraftKings", "line": 6.5, "prediction": 7.87, "side": "Over", "odds": 100, "recommendation": 0, "ev": 3.18, "kelly": 0.318, "sigma": "Med"},
    {"name": "Tyus Jones", "bookmaker": "BetMGM", "line": 3.5, "prediction": 4.46, "side": "Over", "odds": -118, "recommendation": 0, "ev": 3.02, "kelly": 0.356, "sigma": "Low"},
];const prizepicksPairsData = [
    {"name1": "Bobby Portis", "name2": "Will Richard", "line1": 14.5, "line2": 6.5, "prediction1": 9.42, "prediction2": 11.1, "side1": "under", "side2": "over", "recommendation": 1, "ev": 6.75, "kelly": 0.338, "sigma1": "High", "sigma2": "High", "hitRate1": 85.4, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 59.6, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "Deni Avdija", "name2": "Will Richard", "line1": 23.5, "line2": 6.5, "prediction1": 29.23, "prediction2": 11.1, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.26, "kelly": 0.313, "sigma1": "High", "sigma2": "High", "hitRate1": 75.4, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 59.6, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "Bobby Portis", "name2": "Deni Avdija", "line1": 14.5, "line2": 23.5, "prediction1": 9.42, "prediction2": 29.23, "side1": "under", "side2": "over", "recommendation": 1, "ev": 6.11, "kelly": 0.305, "sigma1": "High", "sigma2": "High", "hitRate1": 85.4, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 75.4, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Tyus Jones", "name2": "Donovan Clingan", "line1": 3.5, "line2": 8.5, "prediction1": 4.46, "prediction2": 11.46, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.57, "kelly": 0.179, "sigma1": "Low", "sigma2": "High", "hitRate1": 4.4, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 73.4, "l5_2": 1.0, "l15_2": 0.6},
    {"name1": "Kobe Sanders", "name2": "Donovan Clingan", "line1": 9.5, "line2": 8.5, "prediction1": 12.7, "prediction2": 11.46, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.28, "kelly": 0.164, "sigma1": "High", "sigma2": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 73.4, "l5_2": 1.0, "l15_2": 0.6},
    {"name1": "Kobe Sanders", "name2": "Buddy Hield", "line1": 9.5, "line2": 7.5, "prediction1": 12.7, "prediction2": 9.95, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.21, "kelly": 0.161, "sigma1": "High", "sigma2": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 33.2, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Tyus Jones", "name2": "Buddy Hield", "line1": 3.5, "line2": 7.5, "prediction1": 4.46, "prediction2": 9.95, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.13, "kelly": 0.157, "sigma1": "Low", "sigma2": "High", "hitRate1": 4.4, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 33.2, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Goga Bitadze", "name2": "Brandin Podziemski", "line1": 4.5, "line2": 11.5, "prediction1": 5.94, "prediction2": 14.85, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.11, "kelly": 0.155, "sigma1": "Low", "sigma2": "High", "hitRate1": 55.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 48.2, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "Nicolas Batum", "name2": "Brandin Podziemski", "line1": 5.0, "line2": 11.5, "prediction1": 6.35, "prediction2": 14.85, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.9, "kelly": 0.145, "sigma1": "Med", "sigma2": "High", "hitRate1": 46.5, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 48.2, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "Nicolas Batum", "name2": "Goga Bitadze", "line1": 5.0, "line2": 4.5, "prediction1": 6.35, "prediction2": 5.94, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.83, "kelly": 0.142, "sigma1": "Med", "sigma2": "Low", "hitRate1": 46.5, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 55.6, "l5_2": 0.8, "l15_2": 0.6},
];const prizepicksTriosData = [
    {"name1": "Bobby Portis", "name2": "Deni Avdija", "name3": "Will Richard", "line1": 14.5, "line2": 23.5, "line3": 6.5, "prediction1": 9.42, "prediction2": 29.23, "prediction3": 11.1, "side1": "under", "side2": "over", "side3": "over", "recommendation": 1, "ev": 13.02, "kelly": 0.26, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 85.4, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 75.4, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 59.6, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Bobby Portis", "name2": "Donovan Clingan", "name3": "Will Richard", "line1": 14.5, "line2": 8.5, "line3": 6.5, "prediction1": 9.42, "prediction2": 11.46, "prediction3": 11.1, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.24, "kelly": 0.225, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 85.4, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 73.4, "l5_2": 1.0, "l15_2": 0.6, "hitRate3": 59.6, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Tyus Jones", "name2": "Deni Avdija", "name3": "Donovan Clingan", "line1": 3.5, "line2": 23.5, "line3": 8.5, "prediction1": 4.46, "prediction2": 29.23, "prediction3": 11.46, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.4, "kelly": 0.168, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 4.4, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 75.4, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 73.4, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Kobe Sanders", "name2": "Tyus Jones", "name3": "Brandin Podziemski", "line1": 9.5, "line2": 3.5, "line3": 11.5, "prediction1": 12.7, "prediction2": 4.46, "prediction3": 14.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.61, "kelly": 0.132, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 4.4, "l5_2": 0.2, "l15_2": 0.27, "hitRate3": 48.2, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Kobe Sanders", "name2": "Goga Bitadze", "name3": "Brandin Podziemski", "line1": 9.5, "line2": 4.5, "line3": 11.5, "prediction1": 12.7, "prediction2": 5.94, "prediction3": 14.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.38, "kelly": 0.128, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 55.6, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 48.2, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Nicolas Batum", "name2": "Goga Bitadze", "name3": "Buddy Hield", "line1": 5.0, "line2": 4.5, "line3": 7.5, "prediction1": 6.35, "prediction2": 5.94, "prediction3": 9.95, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.82, "kelly": 0.116, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "hitRate1": 46.5, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 55.6, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 33.2, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Nicolas Batum", "name2": "Zach Edey", "name3": "Buddy Hield", "line1": 5.0, "line2": 12.5, "line3": 7.5, "prediction1": 6.35, "prediction2": 14.88, "prediction3": 9.95, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.09, "kelly": 0.102, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "hitRate1": 46.5, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 22.0, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 33.2, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Brook Lopez", "name2": "Zach Edey", "name3": "Al Horford", "line1": 6.5, "line2": 12.5, "line3": 6.5, "prediction1": 7.87, "prediction2": 14.88, "prediction3": 7.93, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 3.67, "kelly": 0.073, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "hitRate1": 47.0, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 22.0, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 36.8, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Brook Lopez", "name2": "Moses Moody", "name3": "Al Horford", "line1": 6.5, "line2": 12.5, "line3": 6.5, "prediction1": 7.87, "prediction2": 15.06, "prediction3": 7.93, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 3.45, "kelly": 0.069, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "hitRate1": 47.0, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 50.1, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 36.8, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Myles Turner", "name2": "VJ Edgecombe", "name3": "Moses Moody", "line1": 16.5, "line2": 14.5, "line3": 12.5, "prediction1": 13.7, "prediction2": 17.21, "prediction3": 15.06, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 3.21, "kelly": 0.064, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 64.6, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 31.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 50.1, "l5_3": 0.4, "l15_3": 0.4},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Bobby Portis", "name2": "Will Richard", "line1": 14.5, "line2": 6.5, "prediction1": 9.42, "prediction2": 11.1, "side1": "under", "side2": "over", "recommendation": 1, "ev": 6.74, "kelly": 0.337, "sigma1": "High", "sigma2": "High", "hitRate1": 85.4, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 59.6, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "Deni Avdija", "name2": "Will Richard", "line1": 23.5, "line2": 6.5, "prediction1": 29.23, "prediction2": 11.1, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.35, "kelly": 0.318, "sigma1": "High", "sigma2": "High", "hitRate1": 75.4, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 59.6, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "Bobby Portis", "name2": "Deni Avdija", "line1": 14.5, "line2": 23.5, "prediction1": 9.42, "prediction2": 29.23, "side1": "under", "side2": "over", "recommendation": 1, "ev": 6.01, "kelly": 0.301, "sigma1": "High", "sigma2": "High", "hitRate1": 85.4, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 75.4, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Tyus Jones", "name2": "Dominick Barlow", "line1": 3.5, "line2": 5.5, "prediction1": 4.46, "prediction2": 8.15, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.24, "kelly": 0.212, "sigma1": "Low", "sigma2": "Med", "hitRate1": 4.4, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 74.1, "l5_2": 0.6, "l15_2": 0.2},
    {"name1": "Kobe Sanders", "name2": "Dominick Barlow", "line1": 9.5, "line2": 5.5, "prediction1": 12.7, "prediction2": 8.15, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.08, "kelly": 0.204, "sigma1": "High", "sigma2": "Med", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 74.1, "l5_2": 0.6, "l15_2": 0.2},
    {"name1": "Tyus Jones", "name2": "Donovan Clingan", "line1": 3.5, "line2": 8.5, "prediction1": 4.46, "prediction2": 11.46, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.63, "kelly": 0.182, "sigma1": "Low", "sigma2": "High", "hitRate1": 4.4, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 73.4, "l5_2": 1.0, "l15_2": 0.6},
    {"name1": "Kobe Sanders", "name2": "Donovan Clingan", "line1": 9.5, "line2": 8.5, "prediction1": 12.7, "prediction2": 11.46, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.48, "kelly": 0.174, "sigma1": "High", "sigma2": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 73.4, "l5_2": 1.0, "l15_2": 0.6},
    {"name1": "Cam Spencer", "name2": "Buddy Hield", "line1": 12.5, "line2": 7.5, "prediction1": 9.37, "prediction2": 9.95, "side1": "under", "side2": "over", "recommendation": 0, "ev": 2.72, "kelly": 0.136, "sigma1": "High", "sigma2": "High", "hitRate1": 79.4, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 33.2, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Cam Spencer", "name2": "Brandin Podziemski", "line1": 12.5, "line2": 11.5, "prediction1": 9.37, "prediction2": 14.85, "side1": "under", "side2": "over", "recommendation": 0, "ev": 2.66, "kelly": 0.133, "sigma1": "High", "sigma2": "High", "hitRate1": 79.4, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 48.2, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "Zach Edey", "name2": "Brandin Podziemski", "line1": 12.5, "line2": 11.5, "prediction1": 14.88, "prediction2": 14.85, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.45, "kelly": 0.122, "sigma1": "Med", "sigma2": "High", "hitRate1": 22.0, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 48.2, "l5_2": 0.4, "l15_2": 0.53},
];const underdogTriosData = [
    {"name1": "Bobby Portis", "name2": "Deni Avdija", "name3": "Will Richard", "line1": 14.5, "line2": 23.5, "line3": 6.5, "prediction1": 9.42, "prediction2": 29.23, "prediction3": 11.1, "side1": "under", "side2": "over", "side3": "over", "recommendation": 1, "ev": 13.02, "kelly": 0.26, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 85.4, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 75.4, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 59.6, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Bobby Portis", "name2": "Dominick Barlow", "name3": "Will Richard", "line1": 14.5, "line2": 5.5, "line3": 6.5, "prediction1": 9.42, "prediction2": 8.15, "prediction3": 11.1, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 12.14, "kelly": 0.243, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 85.4, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 74.1, "l5_2": 0.6, "l15_2": 0.2, "hitRate3": 59.6, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Dominick Barlow", "name2": "Deni Avdija", "name3": "Donovan Clingan", "line1": 5.5, "line2": 23.5, "line3": 8.5, "prediction1": 8.15, "prediction2": 29.23, "prediction3": 11.46, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.23, "kelly": 0.185, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 74.1, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 75.4, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 73.4, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Tyus Jones", "name2": "Kobe Sanders", "name3": "Donovan Clingan", "line1": 3.5, "line2": 9.5, "line3": 8.5, "prediction1": 4.46, "prediction2": 12.7, "prediction3": 11.46, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.92, "kelly": 0.138, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 4.4, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 16.8, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 73.4, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Tyus Jones", "name2": "Kobe Sanders", "name3": "Brandin Podziemski", "line1": 3.5, "line2": 9.5, "line3": 11.5, "prediction1": 4.46, "prediction2": 12.7, "prediction3": 14.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.61, "kelly": 0.132, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 4.4, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 16.8, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 48.2, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Cam Spencer", "name2": "Brandin Podziemski", "name3": "Buddy Hield", "line1": 12.5, "line2": 11.5, "line3": 7.5, "prediction1": 9.37, "prediction2": 14.85, "prediction3": 9.95, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.67, "kelly": 0.113, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 79.4, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 48.2, "l5_2": 0.4, "l15_2": 0.53, "hitRate3": 33.2, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Zach Edey", "name2": "Cam Spencer", "name3": "Buddy Hield", "line1": 12.5, "line2": 12.5, "line3": 7.5, "prediction1": 14.88, "prediction2": 9.37, "prediction3": 9.95, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 4.97, "kelly": 0.099, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 22.0, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 79.4, "l5_2": 0.2, "l15_2": 0.2, "hitRate3": 33.2, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Zach Edey", "name2": "VJ Edgecombe", "name3": "Al Horford", "line1": 12.5, "line2": 14.5, "line3": 6.5, "prediction1": 14.88, "prediction2": 17.21, "prediction3": 7.93, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 3.65, "kelly": 0.073, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "hitRate1": 22.0, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 31.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 36.8, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "VJ Edgecombe", "name2": "Al Horford", "name3": "Moses Moody", "line1": 14.5, "line2": 6.5, "line3": 12.5, "prediction1": 17.21, "prediction2": 7.93, "prediction3": 15.06, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 3.43, "kelly": 0.069, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 31.1, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 36.8, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 50.1, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Santi Aldama", "name2": "Jrue Holiday", "name3": "Moses Moody", "line1": 16.5, "line2": 13.5, "line3": 12.5, "prediction1": 13.8, "prediction2": 15.91, "prediction3": 15.06, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 2.93, "kelly": 0.059, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 75.2, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 58.8, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 50.1, "l5_3": 0.4, "l15_3": 0.4},
];const prizepicksPointsHitRates = [
    {"name": "Onyeka Okongwu", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.826, "underPct": 0.174},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.803, "underPct": 0.197},
    {"name": "Trendon Watford", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.763, "underPct": 0.237},
    {"name": "Deni Avdija", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.754, "underPct": 0.246},
    {"name": "Donovan Clingan", "line": 8.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.734, "underPct": 0.266},
    {"name": "Stephen Curry", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.688, "underPct": 0.312},
    {"name": "Paul George", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.618, "underPct": 0.382},
    {"name": "Quentin Grimes", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.614, "underPct": 0.386},
    {"name": "Nickeil Alexander-Walker", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.607, "underPct": 0.393},
    {"name": "Harrison Barnes", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.607, "underPct": 0.393},
    {"name": "Will Richard", "line": 6.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.596, "underPct": 0.404},
    {"name": "Jrue Holiday", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.586, "underPct": 0.414},
    {"name": "Goga Bitadze", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.556, "underPct": 0.444},
    {"name": "Anthony Black", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.527, "underPct": 0.473},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.505, "underPct": 0.495},
    {"name": "James Harden", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.501, "underPct": 0.499},
    {"name": "Julian Champagnie", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.497, "underPct": 0.503},
    {"name": "Jalen Johnson", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.495, "underPct": 0.505},
    {"name": "Tyrese Maxey", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.494, "underPct": 0.506},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.483, "underPct": 0.517},
    {"name": "Brandin Podziemski", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.482, "underPct": 0.518},
    {"name": "Brook Lopez", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.47, "underPct": 0.53},
    {"name": "Nicolas Batum", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.465, "underPct": 0.535},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Kris Dunn", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.42, "underPct": 0.58},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.42, "underPct": 0.58},
    {"name": "Malik Monk", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.416, "underPct": 0.584},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.405, "underPct": 0.595},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.377, "underPct": 0.623},
    {"name": "Al Horford", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.368, "underPct": 0.632},
    {"name": "Luke Kornet", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.366, "underPct": 0.634},
    {"name": "Tristan da Silva", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.365, "underPct": 0.635},
    {"name": "Jalen Suggs", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.364, "underPct": 0.636},
    {"name": "Myles Turner", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.354, "underPct": 0.646},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.347, "underPct": 0.653},
    {"name": "Desmond Bane", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.347, "underPct": 0.653},
    {"name": "Kyle Kuzma", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.333, "underPct": 0.667},
    {"name": "Buddy Hield", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.332, "underPct": 0.668},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.311, "underPct": 0.689},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.294, "underPct": 0.706},
    {"name": "Jaylen Wells", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.283, "underPct": 0.717},
    {"name": "Ryan Rollins", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.279, "underPct": 0.721},
    {"name": "Santi Aldama", "line": 16.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.248, "underPct": 0.752},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.22, "underPct": 0.78},
    {"name": "John Collins", "line": 14.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.196, "underPct": 0.804},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.189, "underPct": 0.811},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.175, "underPct": 0.825},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.17, "underPct": 0.83},
    {"name": "Kobe Sanders", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.168, "underPct": 0.832},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.154, "underPct": 0.846},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.146, "underPct": 0.854},
    {"name": "Kentavious Caldwell-Pope", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.145, "underPct": 0.855},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.116, "underPct": 0.884},
    {"name": "Cole Anthony", "line": 10.5, "l5": 0.0, "l10": 0.4, "l15": 0.4, "overPct": 0.106, "underPct": 0.894},
    {"name": "Keldon Johnson", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.102, "underPct": 0.898},
    {"name": "Tyus Jones", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.044, "underPct": 0.956},
    {"name": "Kelly Olynyk", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.031, "underPct": 0.969},
];const prizepicksAssistsHitRates = [
    {"name": "Paul George", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.832, "underPct": 0.168},
    {"name": "Jrue Holiday", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.739, "underPct": 0.261},
    {"name": "Cole Anthony", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.594, "underPct": 0.406},
    {"name": "Al Horford", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.593, "underPct": 0.407},
    {"name": "VJ Edgecombe", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.514, "underPct": 0.486},
    {"name": "Buddy Hield", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.51, "underPct": 0.49},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.442, "underPct": 0.558},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.438, "underPct": 0.562},
    {"name": "James Harden", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.432, "underPct": 0.568},
    {"name": "Anthony Black", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Russell Westbrook", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.427, "underPct": 0.573},
    {"name": "Franz Wagner", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.425, "underPct": 0.575},
    {"name": "Cedric Coward", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Kelly Olynyk", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.408, "underPct": 0.592},
    {"name": "Draymond Green", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.383, "underPct": 0.617},
    {"name": "Kyle Kuzma", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.361, "underPct": 0.639},
    {"name": "Jalen Suggs", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.343, "underPct": 0.657},
    {"name": "Tristan da Silva", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.247, "underPct": 0.753},
    {"name": "Zach Edey", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.239, "underPct": 0.761},
    {"name": "Ryan Rollins", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.235, "underPct": 0.765},
];const prizepicksReboundsHitRates = [
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.628, "underPct": 0.372},
    {"name": "Keldon Johnson", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.625, "underPct": 0.375},
    {"name": "Jrue Holiday", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Franz Wagner", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.584, "underPct": 0.416},
    {"name": "Jock Landale", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.559, "underPct": 0.441},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.546, "underPct": 0.454},
    {"name": "Cedric Coward", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "James Harden", "line": 6.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "VJ Edgecombe", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.516, "underPct": 0.484},
    {"name": "Ivica Zubac", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "Zach LaVine", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 9.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.487, "underPct": 0.513},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.487, "underPct": 0.513},
    {"name": "Al Horford", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.445, "underPct": 0.555},
    {"name": "Zach Edey", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.437, "underPct": 0.563},
    {"name": "Shaedon Sharpe", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.417, "underPct": 0.583},
    {"name": "Toumani Camara", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.412, "underPct": 0.588},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.408, "underPct": 0.592},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.407, "underPct": 0.593},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.392, "underPct": 0.608},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.374, "underPct": 0.626},
    {"name": "Moses Moody", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.37, "underPct": 0.63},
    {"name": "Ryan Rollins", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.352, "underPct": 0.648},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.347, "underPct": 0.653},
    {"name": "Will Richard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.336, "underPct": 0.664},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.321, "underPct": 0.679},
    {"name": "John Collins", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.297, "underPct": 0.703},
    {"name": "Goga Bitadze", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.294, "underPct": 0.706},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.288, "underPct": 0.712},
    {"name": "Harrison Barnes", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.262, "underPct": 0.738},
    {"name": "Myles Turner", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.259, "underPct": 0.741},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.244, "underPct": 0.756},
    {"name": "Kelly Olynyk", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.242, "underPct": 0.758},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.236, "underPct": 0.764},
    {"name": "Kobe Sanders", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.229, "underPct": 0.771},
    {"name": "Stephen Curry", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.222, "underPct": 0.778},
    {"name": "Brook Lopez", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.208, "underPct": 0.792},
    {"name": "Devin Vassell", "line": 4.0, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.203, "underPct": 0.797},
    {"name": "Luke Kornet", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.173, "underPct": 0.827},
    {"name": "Julian Champagnie", "line": 4.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.172, "underPct": 0.828},
    {"name": "Bobby Portis", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.171, "underPct": 0.829},
    {"name": "Jeremy Sochan", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.13, "underPct": 0.87},
];const prizepicksBlocksHitRates = [
    {"name": "John Collins", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.349, "underPct": 0.651},
    {"name": "Jalen Suggs", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.332, "underPct": 0.668},
    {"name": "Brook Lopez", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.478, "underPct": 0.522},
    {"name": "Kyle Kuzma", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.411, "underPct": 0.589},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
];const prizepicksStealsHitRates = [
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyus Jones", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.587, "underPct": 0.413},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.413, "underPct": 0.587},
    {"name": "Jeremy Sochan", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.319, "underPct": 0.681},
    {"name": "Zach LaVine", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Jock Landale", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.611, "underPct": 0.389},
    {"name": "Bobby Portis", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.329, "underPct": 0.671},
    {"name": "Donovan Clingan", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Shaedon Sharpe", "line": 32.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jrue Holiday", "line": 25.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Johnson", "line": 38.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Al Horford", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Justin Edwards", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 41.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nicolas Batum", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Malik Monk", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 36.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandin Podziemski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kobe Sanders", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 37.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keldon Johnson", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 20.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kelly Olynyk", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Spencer", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyus Jones", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Vassell", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jeremy Sochan", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cedric Coward", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Santi Aldama", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Myles Turner", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kentavious Caldwell-Pope", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julian Champagnie", "line": 16.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bobby Portis", "line": 24.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Zach Edey", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "John Collins", "line": 20.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cole Anthony", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const prizepicksPRHitRates = [
    {"name": "James Harden", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jrue Holiday", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shaedon Sharpe", "line": 30.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nicolas Batum", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trendon Watford", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandin Podziemski", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ivica Zubac", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kobe Sanders", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Dunn", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brook Lopez", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremy Sochan", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Paul George", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Julian Champagnie", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kelly Olynyk", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cedric Coward", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Santi Aldama", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Wells", "line": 15.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cole Anthony", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bobby Portis", "line": 22.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "John Collins", "line": 19.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 22.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPAHitRates = [
    {"name": "Al Horford", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jrue Holiday", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephen Curry", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luke Kornet", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Black", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brook Lopez", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kobe Sanders", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kris Dunn", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Buddy Hield", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kelly Olynyk", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "De'Aaron Fox", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cole Anthony", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jeremy Sochan", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keldon Johnson", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paul George", "line": 14.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Bobby Portis", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cedric Coward", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Zach Edey", "line": 14.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksRAHitRates = [
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jock Landale", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cole Anthony", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Rollins", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tristan da Silva", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keldon Johnson", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyus Jones", "line": 3.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Quentin Grimes", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 13.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 11.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Desmond Bane", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trendon Watford", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Drew Eubanks", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Nickeil Alexander-Walker", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kelly Olynyk", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Vassell", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 9.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cam Spencer", "line": 7.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Julian Champagnie", "line": 5.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
];const prizepicksTurnoversHitRates = [
    {"name": "Justin Edwards", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nicolas Batum", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brook Lopez", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Santi Aldama", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Franz Wagner", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "VJ Edgecombe", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 1.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Russell Westbrook", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
];const underdogPointsHitRates = [
    {"name": "Onyeka Okongwu", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.826, "underPct": 0.174},
    {"name": "Trendon Watford", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.763, "underPct": 0.237},
    {"name": "Deni Avdija", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.754, "underPct": 0.246},
    {"name": "Dominick Barlow", "line": 5.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.741, "underPct": 0.259},
    {"name": "Donovan Clingan", "line": 8.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.734, "underPct": 0.266},
    {"name": "Stephen Curry", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.688, "underPct": 0.312},
    {"name": "Paul George", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.618, "underPct": 0.382},
    {"name": "Quentin Grimes", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.614, "underPct": 0.386},
    {"name": "Nickeil Alexander-Walker", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.607, "underPct": 0.393},
    {"name": "Will Richard", "line": 6.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.596, "underPct": 0.404},
    {"name": "Jrue Holiday", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.586, "underPct": 0.414},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.505, "underPct": 0.495},
    {"name": "James Harden", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.501, "underPct": 0.499},
    {"name": "Julian Champagnie", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.497, "underPct": 0.503},
    {"name": "Jalen Johnson", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.495, "underPct": 0.505},
    {"name": "Tyrese Maxey", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.494, "underPct": 0.506},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.483, "underPct": 0.517},
    {"name": "Brandin Podziemski", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.482, "underPct": 0.518},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.42, "underPct": 0.58},
    {"name": "Malik Monk", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.416, "underPct": 0.584},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.405, "underPct": 0.595},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.377, "underPct": 0.623},
    {"name": "Al Horford", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.368, "underPct": 0.632},
    {"name": "Luke Kornet", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.366, "underPct": 0.634},
    {"name": "Tristan da Silva", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.365, "underPct": 0.635},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.347, "underPct": 0.653},
    {"name": "Desmond Bane", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.347, "underPct": 0.653},
    {"name": "Kyle Kuzma", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.333, "underPct": 0.667},
    {"name": "Buddy Hield", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.332, "underPct": 0.668},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.311, "underPct": 0.689},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.294, "underPct": 0.706},
    {"name": "Ryan Rollins", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.279, "underPct": 0.721},
    {"name": "Santi Aldama", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.248, "underPct": 0.752},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.22, "underPct": 0.78},
    {"name": "Cam Spencer", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.206, "underPct": 0.794},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.175, "underPct": 0.825},
    {"name": "Kobe Sanders", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.168, "underPct": 0.832},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.154, "underPct": 0.846},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.146, "underPct": 0.854},
    {"name": "Keldon Johnson", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.102, "underPct": 0.898},
    {"name": "Tyus Jones", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.044, "underPct": 0.956},
];const underdogAssistsHitRates = [
    {"name": "Jrue Holiday", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.739, "underPct": 0.261},
    {"name": "Al Horford", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.593, "underPct": 0.407},
    {"name": "VJ Edgecombe", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.514, "underPct": 0.486},
    {"name": "Buddy Hield", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.51, "underPct": 0.49},
    {"name": "Zach LaVine", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.499, "underPct": 0.501},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.442, "underPct": 0.558},
    {"name": "Anthony Black", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Cedric Coward", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Draymond Green", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.383, "underPct": 0.617},
    {"name": "Kyle Kuzma", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.361, "underPct": 0.639},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.278, "underPct": 0.722},
    {"name": "Tristan da Silva", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.247, "underPct": 0.753},
    {"name": "Zach Edey", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.239, "underPct": 0.761},
    {"name": "Cam Spencer", "line": 4.5, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.122, "underPct": 0.878},
];const underdogReboundsHitRates = [
    {"name": "Keldon Johnson", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.625, "underPct": 0.375},
    {"name": "Jrue Holiday", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.546, "underPct": 0.454},
    {"name": "Al Horford", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.445, "underPct": 0.555},
    {"name": "Zach Edey", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.437, "underPct": 0.563},
    {"name": "Shaedon Sharpe", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.417, "underPct": 0.583},
    {"name": "Toumani Camara", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.412, "underPct": 0.588},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.374, "underPct": 0.626},
    {"name": "Moses Moody", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.37, "underPct": 0.63},
    {"name": "Will Richard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.336, "underPct": 0.664},
    {"name": "John Collins", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.297, "underPct": 0.703},
    {"name": "Kelly Olynyk", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.242, "underPct": 0.758},
    {"name": "Kobe Sanders", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.229, "underPct": 0.771},
    {"name": "Brook Lopez", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.208, "underPct": 0.792},
    {"name": "Julian Champagnie", "line": 4.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.172, "underPct": 0.828},
    {"name": "Bobby Portis", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.171, "underPct": 0.829},
];const underdogBlocksHitRates = [
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
];const underdogStealsHitRates = [
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.413, "underPct": 0.587},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 32.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jrue Holiday", "line": 25.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 38.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Justin Edwards", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Rollins", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Draymond Green", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 41.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jock Landale", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nicolas Batum", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Buddy Hield", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 36.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandin Podziemski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kobe Sanders", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "De'Aaron Fox", "line": 36.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kelly Olynyk", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Malik Monk", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kentavious Caldwell-Pope", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Santi Aldama", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremy Sochan", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Paul George", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Myles Turner", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Vassell", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 20.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julian Champagnie", "line": 16.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bobby Portis", "line": 24.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPRHitRates = [
    {"name": "Shaedon Sharpe", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Johnson", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keldon Johnson", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ivica Zubac", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cedric Coward", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "De'Aaron Fox", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const underdogPAHitRates = [
    {"name": "Shaedon Sharpe", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jrue Holiday", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Franz Wagner", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "De'Aaron Fox", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const underdogRAHitRates = [
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jock Landale", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jrue Holiday", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "DeMar DeRozan", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Santi Aldama", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kyle Kuzma", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Spencer", "line": 7.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Myles Turner", "line": 9.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const underdogTurnoversHitRates = [
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
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

