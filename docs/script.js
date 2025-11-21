const prizepicksSinglesData = [
    {"name": "Justin Edwards", "bookmaker": "DraftKings", "line": 6.5, "prediction": 9.93, "side": "Over", "odds": 100, "recommendation": 0, "ev": 4.94, "kelly": 0.494, "sigma": "High"},
    {"name": "Tristan da Silva", "bookmaker": "BetRivers", "line": 13.5, "prediction": 17.13, "side": "Over", "odds": 114, "recommendation": 0, "ev": 4.93, "kelly": 0.432, "sigma": "High"},
    {"name": "Tristan da Silva", "bookmaker": "BetRivers", "line": 12.5, "prediction": 17.13, "side": "Over", "odds": -104, "recommendation": 1, "ev": 4.71, "kelly": 0.49, "sigma": "High"},
    {"name": "Tristan da Silva", "bookmaker": "BetMGM", "line": 12.5, "prediction": 17.13, "side": "Over", "odds": -105, "recommendation": 1, "ev": 4.46, "kelly": 0.468, "sigma": "High"},
    {"name": "Justin Edwards", "bookmaker": "BetMGM", "line": 6.5, "prediction": 9.93, "side": "Over", "odds": -115, "recommendation": 0, "ev": 4.18, "kelly": 0.48, "sigma": "High"},
    {"name": "Tristan da Silva", "bookmaker": "BetRivers", "line": 11.5, "prediction": 17.13, "side": "Over", "odds": -124, "recommendation": 1, "ev": 4.17, "kelly": 0.517, "sigma": "High"},
    {"name": "Tristan da Silva", "bookmaker": "FanDuel", "line": 11.5, "prediction": 17.13, "side": "Over", "odds": -125, "recommendation": 1, "ev": 4.16, "kelly": 0.52, "sigma": "High"},
    {"name": "Tristan da Silva", "bookmaker": "DraftKings", "line": 11.5, "prediction": 17.13, "side": "Over", "odds": -124, "recommendation": 1, "ev": 4.14, "kelly": 0.513, "sigma": "High"},
    {"name": "Justin Edwards", "bookmaker": "FanDuel", "line": 5.5, "prediction": 9.93, "side": "Over", "odds": -132, "recommendation": 0, "ev": 4.12, "kelly": 0.544, "sigma": "High"},
    {"name": "Tristan da Silva", "bookmaker": "BetRivers", "line": 10.5, "prediction": 17.13, "side": "Over", "odds": -152, "recommendation": 1, "ev": 3.73, "kelly": 0.566, "sigma": "High"},
    {"name": "Zaccharie Risacher", "bookmaker": "BetRivers", "line": 12.5, "prediction": 15.56, "side": "Over", "odds": 100, "recommendation": 0, "ev": 3.6, "kelly": 0.36, "sigma": "High"},
    {"name": "Zaccharie Risacher", "bookmaker": "DraftKings", "line": 11.5, "prediction": 15.56, "side": "Over", "odds": -121, "recommendation": 0, "ev": 3.47, "kelly": 0.42, "sigma": "High"},
    {"name": "Zaccharie Risacher", "bookmaker": "FanDuel", "line": 11.5, "prediction": 15.56, "side": "Over", "odds": -120, "recommendation": 0, "ev": 3.36, "kelly": 0.403, "sigma": "High"},
    {"name": "Bobby Portis", "bookmaker": "FanDuel", "line": 14.5, "prediction": 11.54, "side": "Under", "odds": -104, "recommendation": 0, "ev": 3.14, "kelly": 0.326, "sigma": "High"},
    {"name": "Zaccharie Risacher", "bookmaker": "BetMGM", "line": 11.5, "prediction": 15.56, "side": "Over", "odds": -125, "recommendation": 0, "ev": 3.11, "kelly": 0.389, "sigma": "High"},
];const prizepicksPairsData = [
    {"name1": "Tristan da Silva", "name2": "Justin Edwards", "line1": 11.5, "line2": 6.5, "prediction1": 17.13, "prediction2": 9.93, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.11, "kelly": 0.306, "sigma1": "High", "sigma2": "High", "hitRate1": 47.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 64.4, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Tristan da Silva", "name2": "Zaccharie Risacher", "line1": 11.5, "line2": 11.5, "prediction1": 17.13, "prediction2": 15.56, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.68, "kelly": 0.284, "sigma1": "High", "sigma2": "High", "hitRate1": 47.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 80.3, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Onyeka Okongwu", "name2": "Justin Edwards", "line1": 12.5, "line2": 6.5, "prediction1": 17.13, "prediction2": 9.93, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.85, "kelly": 0.242, "sigma1": "High", "sigma2": "High", "hitRate1": 92.7, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 64.4, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Onyeka Okongwu", "name2": "Myles Turner", "line1": 12.5, "line2": 16.5, "prediction1": 17.13, "prediction2": 12.97, "side1": "over", "side2": "under", "recommendation": 0, "ev": 3.62, "kelly": 0.181, "sigma1": "High", "sigma2": "High", "hitRate1": 92.7, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 64.6, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Zaccharie Risacher", "name2": "Myles Turner", "line1": 11.5, "line2": 16.5, "prediction1": 15.56, "prediction2": 12.97, "side1": "over", "side2": "under", "recommendation": 0, "ev": 3.5, "kelly": 0.175, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 64.6, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Luke Kennard", "name2": "Russell Westbrook", "line1": 5.5, "line2": 13.5, "prediction1": 6.97, "prediction2": 16.61, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.43, "kelly": 0.121, "sigma1": "Med", "sigma2": "High", "hitRate1": 82.4, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 34.7, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Goga Bitadze", "name2": "Russell Westbrook", "line1": 4.5, "line2": 13.5, "prediction1": 5.74, "prediction2": 16.61, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.4, "kelly": 0.12, "sigma1": "Low", "sigma2": "High", "hitRate1": 55.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 34.7, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Goga Bitadze", "name2": "Nickeil Alexander-Walker", "line1": 4.5, "line2": 16.5, "prediction1": 5.74, "prediction2": 19.91, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.39, "kelly": 0.12, "sigma1": "Low", "sigma2": "High", "hitRate1": 55.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 69.4, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Nickeil Alexander-Walker", "name2": "Bobby Portis", "line1": 16.5, "line2": 14.5, "prediction1": 19.91, "prediction2": 11.54, "side1": "over", "side2": "under", "recommendation": 0, "ev": 2.21, "kelly": 0.11, "sigma1": "High", "sigma2": "High", "hitRate1": 69.4, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 85.4, "l5_2": 0.0, "l15_2": 0.07},
    {"name1": "Luke Kennard", "name2": "Bobby Portis", "line1": 5.5, "line2": 14.5, "prediction1": 6.97, "prediction2": 11.54, "side1": "over", "side2": "under", "recommendation": 0, "ev": 2.09, "kelly": 0.105, "sigma1": "Med", "sigma2": "High", "hitRate1": 82.4, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 85.4, "l5_2": 0.0, "l15_2": 0.07},
];const prizepicksTriosData = [
    {"name1": "Tristan da Silva", "name2": "Onyeka Okongwu", "name3": "Justin Edwards", "line1": 11.5, "line2": 12.5, "line3": 6.5, "prediction1": 17.13, "prediction2": 17.13, "prediction3": 9.93, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.42, "kelly": 0.228, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 47.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 92.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 64.4, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "Tristan da Silva", "name2": "Zaccharie Risacher", "name3": "Justin Edwards", "line1": 11.5, "line2": 11.5, "line3": 6.5, "prediction1": 17.13, "prediction2": 15.56, "prediction3": 9.93, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.08, "kelly": 0.222, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 47.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 80.3, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 64.4, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "Goga Bitadze", "name2": "Onyeka Okongwu", "name3": "Zaccharie Risacher", "line1": 4.5, "line2": 12.5, "line3": 11.5, "prediction1": 5.74, "prediction2": 17.13, "prediction3": 15.56, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.03, "kelly": 0.161, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 55.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 92.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 80.3, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Goga Bitadze", "name2": "Nickeil Alexander-Walker", "name3": "Myles Turner", "line1": 4.5, "line2": 16.5, "line3": 16.5, "prediction1": 5.74, "prediction2": 19.91, "prediction3": 12.97, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 5.07, "kelly": 0.101, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 55.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 69.4, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 64.6, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Nickeil Alexander-Walker", "name2": "Luke Kennard", "name3": "Myles Turner", "line1": 16.5, "line2": 5.5, "line3": 16.5, "prediction1": 19.91, "prediction2": 6.97, "prediction3": 12.97, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 4.97, "kelly": 0.099, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 69.4, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 82.4, "l5_2": 0.6, "l15_2": 0.67, "hitRate3": 64.6, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Luke Kennard", "name2": "Russell Westbrook", "name3": "Bobby Portis", "line1": 5.5, "line2": 13.5, "line3": 14.5, "prediction1": 6.97, "prediction2": 16.61, "prediction3": 11.54, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 4.66, "kelly": 0.093, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 82.4, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 34.7, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 85.4, "l5_3": 0.0, "l15_3": 0.07},
    {"name1": "Santi Aldama", "name2": "Russell Westbrook", "name3": "Bobby Portis", "line1": 18.5, "line2": 13.5, "line3": 14.5, "prediction1": 15.56, "prediction2": 16.61, "prediction3": 11.54, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 4.38, "kelly": 0.088, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 88.0, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 34.7, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 85.4, "l5_3": 0.0, "l15_3": 0.07},
    {"name1": "Santi Aldama", "name2": "VJ Edgecombe", "name3": "Kyle Kuzma", "line1": 18.5, "line2": 14.5, "line3": 14.5, "prediction1": 15.56, "prediction2": 17.42, "prediction3": 17.42, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 3.79, "kelly": 0.076, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 88.0, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 31.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 43.2, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Kobe Sanders", "name2": "VJ Edgecombe", "name3": "Kyle Kuzma", "line1": 9.5, "line2": 14.5, "line3": 14.5, "prediction1": 11.54, "prediction2": 17.42, "prediction3": 17.42, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 3.59, "kelly": 0.072, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 31.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 43.2, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Kobe Sanders", "name2": "Brook Lopez", "name3": "Jock Landale", "line1": 9.5, "line2": 6.0, "line3": 8.5, "prediction1": 11.54, "prediction2": 6.97, "prediction3": 9.93, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 2.76, "kelly": 0.055, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 47.0, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 50.5, "l5_3": 0.4, "l15_3": 0.6},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Tristan da Silva", "name2": "Onyeka Okongwu", "line1": 11.5, "line2": 13.5, "prediction1": 17.13, "prediction2": 17.13, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.83, "kelly": 0.241, "sigma1": "High", "sigma2": "High", "hitRate1": 47.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 88.4, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Tristan da Silva", "name2": "Luke Kennard", "line1": 11.5, "line2": 5.5, "prediction1": 17.13, "prediction2": 6.97, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.53, "kelly": 0.227, "sigma1": "High", "sigma2": "Med", "hitRate1": 47.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 82.4, "l5_2": 0.6, "l15_2": 0.67},
    {"name1": "Onyeka Okongwu", "name2": "Russell Westbrook", "line1": 13.5, "line2": 13.5, "prediction1": 17.13, "prediction2": 16.61, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.84, "kelly": 0.142, "sigma1": "High", "sigma2": "High", "hitRate1": 88.4, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 34.7, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Nickeil Alexander-Walker", "name2": "Russell Westbrook", "line1": 16.5, "line2": 13.5, "prediction1": 19.91, "prediction2": 16.61, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.27, "kelly": 0.114, "sigma1": "High", "sigma2": "High", "hitRate1": 69.4, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 34.7, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Luke Kennard", "name2": "Bobby Portis", "line1": 5.5, "line2": 14.5, "prediction1": 6.97, "prediction2": 11.54, "side1": "over", "side2": "under", "recommendation": 0, "ev": 2.12, "kelly": 0.106, "sigma1": "Med", "sigma2": "High", "hitRate1": 82.4, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 85.4, "l5_2": 0.0, "l15_2": 0.07},
    {"name1": "Nickeil Alexander-Walker", "name2": "Bobby Portis", "line1": 16.5, "line2": 14.5, "prediction1": 19.91, "prediction2": 11.54, "side1": "over", "side2": "under", "recommendation": 0, "ev": 2.05, "kelly": 0.102, "sigma1": "High", "sigma2": "High", "hitRate1": 69.4, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 85.4, "l5_2": 0.0, "l15_2": 0.07},
    {"name1": "Kobe Sanders", "name2": "VJ Edgecombe", "line1": 9.5, "line2": 14.5, "prediction1": 11.54, "prediction2": 17.42, "side1": "over", "side2": "over", "recommendation": 0, "ev": 1.63, "kelly": 0.081, "sigma1": "Med", "sigma2": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 31.1, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Kyle Kuzma", "name2": "VJ Edgecombe", "line1": 15.5, "line2": 14.5, "prediction1": 17.42, "prediction2": 17.42, "side1": "over", "side2": "over", "recommendation": 0, "ev": 0.86, "kelly": 0.043, "sigma1": "High", "sigma2": "High", "hitRate1": 33.3, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 31.1, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Kobe Sanders", "name2": "Kyle Kuzma", "line1": 9.5, "line2": 15.5, "prediction1": 11.54, "prediction2": 17.42, "side1": "over", "side2": "over", "recommendation": 0, "ev": 0.78, "kelly": 0.039, "sigma1": "Med", "sigma2": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 33.3, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Santi Aldama", "name2": "Trendon Watford", "line1": 17.5, "line2": 7.5, "prediction1": 15.56, "prediction2": 8.32, "side1": "under", "side2": "over", "recommendation": 0, "ev": -0.21, "kelly": 0.0, "sigma1": "High", "sigma2": "Med", "hitRate1": 82.4, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 76.3, "l5_2": 0.4, "l15_2": 0.4},
];const underdogTriosData = [
    {"name1": "Tristan da Silva", "name2": "Onyeka Okongwu", "name3": "Nickeil Alexander-Walker", "line1": 11.5, "line2": 13.5, "line3": 16.5, "prediction1": 17.13, "prediction2": 17.13, "prediction3": 19.91, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.02, "kelly": 0.16, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 47.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 88.4, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 69.4, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Tristan da Silva", "name2": "Onyeka Okongwu", "name3": "Russell Westbrook", "line1": 11.5, "line2": 13.5, "line3": 13.5, "prediction1": 17.13, "prediction2": 17.13, "prediction3": 16.61, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.0, "kelly": 0.16, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 47.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 88.4, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 34.7, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Luke Kennard", "name2": "Nickeil Alexander-Walker", "name3": "Russell Westbrook", "line1": 5.5, "line2": 16.5, "line3": 13.5, "prediction1": 6.97, "prediction2": 19.91, "prediction3": 16.61, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 4.9, "kelly": 0.098, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 82.4, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 69.4, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 34.7, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Luke Kennard", "name2": "VJ Edgecombe", "name3": "Bobby Portis", "line1": 5.5, "line2": 14.5, "line3": 14.5, "prediction1": 6.97, "prediction2": 17.42, "prediction3": 11.54, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 4.28, "kelly": 0.086, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 82.4, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 31.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 85.4, "l5_3": 0.0, "l15_3": 0.07},
    {"name1": "Kobe Sanders", "name2": "VJ Edgecombe", "name3": "Bobby Portis", "line1": 9.5, "line2": 14.5, "line3": 14.5, "prediction1": 11.54, "prediction2": 17.42, "prediction3": 11.54, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 3.8, "kelly": 0.076, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 31.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 85.4, "l5_3": 0.0, "l15_3": 0.07},
    {"name1": "Kobe Sanders", "name2": "Santi Aldama", "name3": "Kyle Kuzma", "line1": 9.5, "line2": 17.5, "line3": 15.5, "prediction1": 11.54, "prediction2": 15.56, "prediction3": 17.42, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 1.61, "kelly": 0.032, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 16.8, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 82.4, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 33.3, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Santi Aldama", "name2": "Kyle Kuzma", "name3": "Trendon Watford", "line1": 17.5, "line2": 15.5, "line3": 7.5, "prediction1": 15.56, "prediction2": 17.42, "prediction3": 8.32, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 0.71, "kelly": 0.014, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 82.4, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 33.3, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 76.3, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Malik Monk", "name2": "Trendon Watford", "name3": "Ryan Rollins", "line1": 13.5, "line2": 7.5, "line3": 19.5, "prediction1": 11.54, "prediction2": 8.32, "prediction3": 17.89, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": -0.02, "kelly": 0.0, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 68.9, "l5_1": 0.4, "l15_1": 0.53, "hitRate2": 76.3, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 64.0, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Keldon Johnson", "name2": "Malik Monk", "name3": "Ryan Rollins", "line1": 14.5, "line2": 13.5, "line3": 19.5, "prediction1": 12.97, "prediction2": 11.54, "prediction3": 17.89, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": -0.43, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 89.8, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 68.9, "l5_2": 0.4, "l15_2": 0.53, "hitRate3": 64.0, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Jalen Johnson", "name2": "Keldon Johnson", "name3": "Quentin Grimes", "line1": 22.5, "line2": 14.5, "line3": 15.5, "prediction1": 23.45, "prediction2": 12.97, "prediction3": 16.52, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": -1.28, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 49.5, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 89.8, "l5_2": 0.2, "l15_2": 0.2, "hitRate3": 61.4, "l5_3": 0.6, "l15_3": 0.53},
];const prizepicksPointsHitRates = [
    {"name": "Onyeka Okongwu", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.927, "underPct": 0.073},
    {"name": "Naji Marshall", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.895, "underPct": 0.105},
    {"name": "Dillon Brooks", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.869, "underPct": 0.131},
    {"name": "Saddiq Bey", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.859, "underPct": 0.141},
    {"name": "Luke Kennard", "line": 5.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.824, "underPct": 0.176},
    {"name": "Trey Murphy III", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.81, "underPct": 0.19},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.803, "underPct": 0.197},
    {"name": "Sandro Mamukelashvili", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.802, "underPct": 0.198},
    {"name": "Aaron Gordon", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.794, "underPct": 0.206},
    {"name": "Jeremiah Fears", "line": 13.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.773, "underPct": 0.227},
    {"name": "Trendon Watford", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.763, "underPct": 0.237},
    {"name": "Grayson Allen", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.758, "underPct": 0.242},
    {"name": "Deni Avdija", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.754, "underPct": 0.246},
    {"name": "Donovan Clingan", "line": 8.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.734, "underPct": 0.266},
    {"name": "Nickeil Alexander-Walker", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.694, "underPct": 0.306},
    {"name": "Stephen Curry", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.688, "underPct": 0.312},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.684, "underPct": 0.316},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.677, "underPct": 0.323},
    {"name": "Jaden McDaniels", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.657, "underPct": 0.343},
    {"name": "Noah Clowney", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.65, "underPct": 0.35},
    {"name": "Justin Edwards", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.644, "underPct": 0.356},
    {"name": "Alperen Sengun", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.642, "underPct": 0.358},
    {"name": "Jakob Poeltl", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.619, "underPct": 0.381},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.617, "underPct": 0.383},
    {"name": "Quentin Grimes", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.614, "underPct": 0.386},
    {"name": "Harrison Barnes", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.607, "underPct": 0.393},
    {"name": "Will Richard", "line": 6.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.596, "underPct": 0.404},
    {"name": "Jrue Holiday", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.586, "underPct": 0.414},
    {"name": "Tre Johnson", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.583, "underPct": 0.417},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.579, "underPct": 0.421},
    {"name": "Luka Garza", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.577, "underPct": 0.423},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.574, "underPct": 0.426},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.565, "underPct": 0.435},
    {"name": "Goga Bitadze", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.556, "underPct": 0.444},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Anthony Black", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.527, "underPct": 0.473},
    {"name": "Jose Alvarado", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.514, "underPct": 0.486},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.511, "underPct": 0.489},
    {"name": "Paul George", "line": 12.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.506, "underPct": 0.494},
    {"name": "Jamal Murray", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.506, "underPct": 0.494},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.505, "underPct": 0.495},
    {"name": "Alex Sarr", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "James Harden", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.501, "underPct": 0.499},
    {"name": "Julian Champagnie", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.497, "underPct": 0.503},
    {"name": "Jalen Johnson", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.495, "underPct": 0.505},
    {"name": "Tyrese Maxey", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.494, "underPct": 0.506},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.483, "underPct": 0.517},
    {"name": "Brandin Podziemski", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.482, "underPct": 0.518},
    {"name": "Tristan da Silva", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.479, "underPct": 0.521},
    {"name": "Brook Lopez", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.47, "underPct": 0.53},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.468, "underPct": 0.532},
    {"name": "Nicolas Batum", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.465, "underPct": 0.535},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.465, "underPct": 0.535},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.444, "underPct": 0.556},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.444, "underPct": 0.556},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.432, "underPct": 0.568},
    {"name": "Kyle Kuzma", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.432, "underPct": 0.568},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.421, "underPct": 0.579},
    {"name": "Kris Dunn", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.42, "underPct": 0.58},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.42, "underPct": 0.58},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.405, "underPct": 0.595},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.399, "underPct": 0.601},
    {"name": "Brandon Ingram", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.397, "underPct": 0.603},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.394, "underPct": 0.606},
    {"name": "Zion Williamson", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.39, "underPct": 0.61},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.384, "underPct": 0.616},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.377, "underPct": 0.623},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.369, "underPct": 0.631},
    {"name": "Al Horford", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.368, "underPct": 0.632},
    {"name": "Luke Kornet", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.366, "underPct": 0.634},
    {"name": "Jalen Suggs", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.364, "underPct": 0.636},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.36, "underPct": 0.64},
    {"name": "Myles Turner", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.354, "underPct": 0.646},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.349, "underPct": 0.651},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.347, "underPct": 0.653},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.344, "underPct": 0.656},
    {"name": "Buddy Hield", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.332, "underPct": 0.668},
    {"name": "D'Angelo Russell", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.321, "underPct": 0.679},
    {"name": "Mark Williams", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.32, "underPct": 0.68},
    {"name": "Khris Middleton", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.315, "underPct": 0.685},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.311, "underPct": 0.689},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.311, "underPct": 0.689},
    {"name": "Ryan Dunn", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.307, "underPct": 0.693},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.294, "underPct": 0.706},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.293, "underPct": 0.707},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.268, "underPct": 0.732},
    {"name": "Cameron Johnson", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.255, "underPct": 0.745},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.253, "underPct": 0.747},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.236, "underPct": 0.764},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.22, "underPct": 0.78},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.219, "underPct": 0.781},
    {"name": "Terance Mann", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.215, "underPct": 0.785},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.21, "underPct": 0.79},
    {"name": "John Collins", "line": 14.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.196, "underPct": 0.804},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.191, "underPct": 0.809},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.189, "underPct": 0.811},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.175, "underPct": 0.825},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.17, "underPct": 0.83},
    {"name": "Kobe Sanders", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.168, "underPct": 0.832},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.154, "underPct": 0.846},
    {"name": "Dereck Lively II", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.146, "underPct": 0.854},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.146, "underPct": 0.854},
    {"name": "Kentavious Caldwell-Pope", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.145, "underPct": 0.855},
    {"name": "Santi Aldama", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.12, "underPct": 0.88},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.118, "underPct": 0.882},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.116, "underPct": 0.884},
    {"name": "Cole Anthony", "line": 10.5, "l5": 0.0, "l10": 0.4, "l15": 0.4, "overPct": 0.106, "underPct": 0.894},
    {"name": "Keldon Johnson", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.102, "underPct": 0.898},
    {"name": "Tyus Jones", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.044, "underPct": 0.956},
    {"name": "Kelly Olynyk", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.031, "underPct": 0.969},
];const prizepicksAssistsHitRates = [
    {"name": "Paul George", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.832, "underPct": 0.168},
    {"name": "Jrue Holiday", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.739, "underPct": 0.261},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.658, "underPct": 0.342},
    {"name": "Alperen Sengun", "line": 6.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.614, "underPct": 0.386},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.612, "underPct": 0.388},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.607, "underPct": 0.393},
    {"name": "Cole Anthony", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.594, "underPct": 0.406},
    {"name": "Al Horford", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.593, "underPct": 0.407},
    {"name": "Terance Mann", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.562, "underPct": 0.438},
    {"name": "VJ Edgecombe", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.514, "underPct": 0.486},
    {"name": "Buddy Hield", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.51, "underPct": 0.49},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Zion Williamson", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.485, "underPct": 0.515},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.442, "underPct": 0.558},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.438, "underPct": 0.562},
    {"name": "James Harden", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.432, "underPct": 0.568},
    {"name": "Max Christie", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.432, "underPct": 0.568},
    {"name": "Anthony Black", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Russell Westbrook", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.427, "underPct": 0.573},
    {"name": "Franz Wagner", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.425, "underPct": 0.575},
    {"name": "Scottie Barnes", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.42, "underPct": 0.58},
    {"name": "Cedric Coward", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Kelly Olynyk", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.408, "underPct": 0.592},
    {"name": "Tyrese Maxey", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.394, "underPct": 0.606},
    {"name": "D'Angelo Russell", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.393, "underPct": 0.607},
    {"name": "Draymond Green", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.383, "underPct": 0.617},
    {"name": "Devin Booker", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.365, "underPct": 0.635},
    {"name": "Kyle Kuzma", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.361, "underPct": 0.639},
    {"name": "Jalen Suggs", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.343, "underPct": 0.657},
    {"name": "Jeremiah Fears", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.337, "underPct": 0.663},
    {"name": "Naji Marshall", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.323, "underPct": 0.677},
    {"name": "De'Aaron Fox", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.296, "underPct": 0.704},
    {"name": "Tristan da Silva", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.247, "underPct": 0.753},
    {"name": "Zach Edey", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.239, "underPct": 0.761},
    {"name": "Ryan Rollins", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.235, "underPct": 0.765},
];const prizepicksReboundsHitRates = [
    {"name": "Alperen Sengun", "line": 9.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.691, "underPct": 0.309},
    {"name": "Trendon Watford", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.689, "underPct": 0.311},
    {"name": "Cedric Coward", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.685, "underPct": 0.315},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.669, "underPct": 0.331},
    {"name": "Brandon Williams", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.644, "underPct": 0.356},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.628, "underPct": 0.372},
    {"name": "Cooper Flagg", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.594, "underPct": 0.406},
    {"name": "Jrue Holiday", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Franz Wagner", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.584, "underPct": 0.416},
    {"name": "Jock Landale", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.559, "underPct": 0.441},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.546, "underPct": 0.454},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.534, "underPct": 0.466},
    {"name": "James Harden", "line": 6.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "P.J. Washington", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.529, "underPct": 0.471},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.516, "underPct": 0.484},
    {"name": "VJ Edgecombe", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.516, "underPct": 0.484},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.514, "underPct": 0.486},
    {"name": "Ivica Zubac", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "Naz Reid", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.499, "underPct": 0.501},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.493, "underPct": 0.507},
    {"name": "Jalen Johnson", "line": 9.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.487, "underPct": 0.513},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.487, "underPct": 0.513},
    {"name": "Payton Pritchard", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.48, "underPct": 0.52},
    {"name": "Onyeka Okongwu", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.476, "underPct": 0.524},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.456, "underPct": 0.544},
    {"name": "Harrison Barnes", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.449, "underPct": 0.551},
    {"name": "Al Horford", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.445, "underPct": 0.555},
    {"name": "Zach Edey", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.437, "underPct": 0.563},
    {"name": "Donte DiVincenzo", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.423, "underPct": 0.577},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.422, "underPct": 0.578},
    {"name": "Shaedon Sharpe", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.417, "underPct": 0.583},
    {"name": "Toumani Camara", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.412, "underPct": 0.588},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.408, "underPct": 0.592},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.407, "underPct": 0.593},
    {"name": "Santi Aldama", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.401, "underPct": 0.599},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.392, "underPct": 0.608},
    {"name": "Anfernee Simons", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.386, "underPct": 0.614},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.374, "underPct": 0.626},
    {"name": "Moses Moody", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.37, "underPct": 0.63},
    {"name": "Luka Garza", "line": 5.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.365, "underPct": 0.635},
    {"name": "Yves Missi", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.363, "underPct": 0.637},
    {"name": "Ryan Dunn", "line": 4.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.361, "underPct": 0.639},
    {"name": "Kevin Durant", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.357, "underPct": 0.643},
    {"name": "Ryan Rollins", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.352, "underPct": 0.648},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.347, "underPct": 0.653},
    {"name": "Will Richard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.336, "underPct": 0.664},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.321, "underPct": 0.679},
    {"name": "Daniel Gafford", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.319, "underPct": 0.681},
    {"name": "John Collins", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.297, "underPct": 0.703},
    {"name": "Goga Bitadze", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.294, "underPct": 0.706},
    {"name": "Myles Turner", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.259, "underPct": 0.741},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.244, "underPct": 0.756},
    {"name": "Kelly Olynyk", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.242, "underPct": 0.758},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.236, "underPct": 0.764},
    {"name": "Mark Williams", "line": 9.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.232, "underPct": 0.768},
    {"name": "Kobe Sanders", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.229, "underPct": 0.771},
    {"name": "Stephen Curry", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.222, "underPct": 0.778},
    {"name": "Brook Lopez", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.208, "underPct": 0.792},
    {"name": "Devin Vassell", "line": 4.0, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.203, "underPct": 0.797},
    {"name": "Kyle Kuzma", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.191, "underPct": 0.809},
    {"name": "Luke Kornet", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.173, "underPct": 0.827},
    {"name": "Julian Champagnie", "line": 4.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.172, "underPct": 0.828},
    {"name": "Bobby Portis", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.171, "underPct": 0.829},
    {"name": "Jeremy Sochan", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.13, "underPct": 0.87},
];const prizepicksBlocksHitRates = [
    {"name": "John Collins", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.349, "underPct": 0.651},
    {"name": "Jalen Suggs", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.332, "underPct": 0.668},
    {"name": "Brook Lopez", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.478, "underPct": 0.522},
    {"name": "Kyle Kuzma", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.411, "underPct": 0.589},
    {"name": "Alex Sarr", "line": 1.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.652, "underPct": 0.348},
    {"name": "Scottie Barnes", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.61, "underPct": 0.39},
    {"name": "Bilal Coulibaly", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.727, "underPct": 0.273},
    {"name": "Cooper Flagg", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.558, "underPct": 0.442},
    {"name": "Amen Thompson", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.394, "underPct": 0.606},
    {"name": "Kevin Durant", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
];const prizepicksStealsHitRates = [
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.531, "underPct": 0.469},
    {"name": "Tyus Jones", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.587, "underPct": 0.413},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.413, "underPct": 0.587},
    {"name": "Jeremy Sochan", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.319, "underPct": 0.681},
    {"name": "Kelly Olynyk", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.336, "underPct": 0.664},
    {"name": "Zach LaVine", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Jock Landale", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.611, "underPct": 0.389},
    {"name": "Drew Eubanks", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.387, "underPct": 0.613},
    {"name": "Day'Ron Sharpe", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.595, "underPct": 0.405},
    {"name": "Neemias Queta", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.45, "underPct": 0.55},
    {"name": "Terance Mann", "line": 0.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.487, "underPct": 0.513},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.364, "underPct": 0.636},
    {"name": "Scottie Barnes", "line": 1.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.425, "underPct": 0.575},
    {"name": "D'Angelo Russell", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.387, "underPct": 0.613},
    {"name": "Daniel Gafford", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.557, "underPct": 0.443},
    {"name": "Dereck Lively II", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.399, "underPct": 0.601},
    {"name": "Saddiq Bey", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.455, "underPct": 0.545},
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.526, "underPct": 0.474},
    {"name": "Donovan Clingan", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Zaccharie Risacher", "line": 15.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luka Garza", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alperen Sengun", "line": 38.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 32.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jrue Holiday", "line": 25.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 34.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Al Horford", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 38.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 37.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dereck Lively II", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "D'Angelo Russell", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Justin Edwards", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 41.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cameron Johnson", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Suggs", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nicolas Batum", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Draymond Green", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 33.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaden McDaniels", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Grayson Allen", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Dunn", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 36.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kelly Olynyk", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Harrison Barnes", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 36.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kobe Sanders", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jeremy Sochan", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Santi Aldama", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cedric Coward", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 40.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paul George", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 18.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kentavious Caldwell-Pope", "line": 14.0, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drew Eubanks", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bobby Portis", "line": 24.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Julian Champagnie", "line": 16.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cole Anthony", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Amen Thompson", "line": 28.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 20.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 1.0, "l10": 0.9, "l15": 0.87, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "James Harden", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 14.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Alperen Sengun", "line": 31.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 19.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shaedon Sharpe", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jrue Holiday", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ziaire Williams", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Neemias Queta", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Al Horford", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nicolas Batum", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Johnson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Grayson Allen", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dereck Lively II", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "D'Angelo Russell", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Yves Missi", "line": 11.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Day'Ron Sharpe", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 23.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremy Sochan", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sandro Mamukelashvili", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brook Lopez", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kobe Sanders", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trendon Watford", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Julian Champagnie", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kelly Olynyk", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Gradey Dick", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 25.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cedric Coward", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kentavious Caldwell-Pope", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 22.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Paul George", "line": 16.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Cole Anthony", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 19.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 22.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPAHitRates = [
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luka Garza", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dillon Brooks", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaden McDaniels", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Naz Reid", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Gordon", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jrue Holiday", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Grayson Allen", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Justin Edwards", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cameron Johnson", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 7.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tristan da Silva", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brook Lopez", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kobe Sanders", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kelly Olynyk", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 31.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Royce O'Neale", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Dunn", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zion Williamson", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tre Johnson", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cole Anthony", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Wells", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Spencer", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Santi Aldama", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 35.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dereck Lively II", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jeremy Sochan", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 14.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Cedric Coward", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Paul George", "line": 14.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Bobby Portis", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const prizepicksRAHitRates = [
    {"name": "Saddiq Bey", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cole Anthony", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandon Williams", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luka Garza", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zion Williamson", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 14.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden McDaniels", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tristan da Silva", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Draymond Green", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Yves Missi", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Alex Sarr", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 13.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derrick White", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Immanuel Quickley", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Vassell", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kelly Olynyk", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Nickeil Alexander-Walker", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Drew Eubanks", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kyle Kuzma", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Corey Kispert", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Whitmore", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sandro Mamukelashvili", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 5.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 9.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const prizepicksTurnoversHitRates = [
    {"name": "Justin Edwards", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 1.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Sandro Mamukelashvili", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brook Lopez", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bilal Coulibaly", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Corey Kispert", "line": 0.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Aaron Gordon", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Franz Wagner", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogPointsHitRates = [
    {"name": "Naji Marshall", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.895, "underPct": 0.105},
    {"name": "Onyeka Okongwu", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.884, "underPct": 0.116},
    {"name": "Dillon Brooks", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.869, "underPct": 0.131},
    {"name": "Saddiq Bey", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.859, "underPct": 0.141},
    {"name": "Luke Kennard", "line": 5.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.824, "underPct": 0.176},
    {"name": "Trey Murphy III", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.81, "underPct": 0.19},
    {"name": "Sandro Mamukelashvili", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.802, "underPct": 0.198},
    {"name": "Aaron Gordon", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.794, "underPct": 0.206},
    {"name": "Jeremiah Fears", "line": 13.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.773, "underPct": 0.227},
    {"name": "Trendon Watford", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.763, "underPct": 0.237},
    {"name": "Grayson Allen", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.758, "underPct": 0.242},
    {"name": "Deni Avdija", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.754, "underPct": 0.246},
    {"name": "Donovan Clingan", "line": 8.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.734, "underPct": 0.266},
    {"name": "Nickeil Alexander-Walker", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.694, "underPct": 0.306},
    {"name": "Stephen Curry", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.688, "underPct": 0.312},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.684, "underPct": 0.316},
    {"name": "Derik Queen", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.681, "underPct": 0.319},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.677, "underPct": 0.323},
    {"name": "Jaden McDaniels", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.657, "underPct": 0.343},
    {"name": "Coby White", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.654, "underPct": 0.346},
    {"name": "Noah Clowney", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.65, "underPct": 0.35},
    {"name": "Alperen Sengun", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.642, "underPct": 0.358},
    {"name": "Jakob Poeltl", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.619, "underPct": 0.381},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.617, "underPct": 0.383},
    {"name": "Quentin Grimes", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.614, "underPct": 0.386},
    {"name": "Harrison Barnes", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.607, "underPct": 0.393},
    {"name": "Will Richard", "line": 6.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.596, "underPct": 0.404},
    {"name": "Jrue Holiday", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.586, "underPct": 0.414},
    {"name": "Tre Johnson", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.583, "underPct": 0.417},
    {"name": "Pelle Larsson", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.58, "underPct": 0.42},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.574, "underPct": 0.426},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.565, "underPct": 0.435},
    {"name": "Corey Kispert", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.543, "underPct": 0.457},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.537, "underPct": 0.463},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Jose Alvarado", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.514, "underPct": 0.486},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.511, "underPct": 0.489},
    {"name": "Paul George", "line": 12.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.506, "underPct": 0.494},
    {"name": "Jamal Murray", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.506, "underPct": 0.494},
    {"name": "Norman Powell", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.505, "underPct": 0.495},
    {"name": "Alex Sarr", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "James Harden", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.501, "underPct": 0.499},
    {"name": "Julian Champagnie", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.497, "underPct": 0.503},
    {"name": "Jalen Johnson", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.495, "underPct": 0.505},
    {"name": "Tyrese Maxey", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.494, "underPct": 0.506},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.487, "underPct": 0.513},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.483, "underPct": 0.517},
    {"name": "Brandin Podziemski", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.482, "underPct": 0.518},
    {"name": "Tristan da Silva", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.479, "underPct": 0.521},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.468, "underPct": 0.532},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.465, "underPct": 0.535},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.444, "underPct": 0.556},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.444, "underPct": 0.556},
    {"name": "Bam Adebayo", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.437, "underPct": 0.563},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.432, "underPct": 0.568},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.421, "underPct": 0.579},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.42, "underPct": 0.58},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.411, "underPct": 0.589},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.405, "underPct": 0.595},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.399, "underPct": 0.601},
    {"name": "Brandon Ingram", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.397, "underPct": 0.603},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.394, "underPct": 0.606},
    {"name": "Zion Williamson", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.39, "underPct": 0.61},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.384, "underPct": 0.616},
    {"name": "Simone Fontecchio", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.379, "underPct": 0.621},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.377, "underPct": 0.623},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.369, "underPct": 0.631},
    {"name": "Al Horford", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.368, "underPct": 0.632},
    {"name": "Luke Kornet", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.366, "underPct": 0.634},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.36, "underPct": 0.64},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.349, "underPct": 0.651},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.347, "underPct": 0.653},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.344, "underPct": 0.656},
    {"name": "Anfernee Simons", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.343, "underPct": 0.657},
    {"name": "Kyle Kuzma", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.333, "underPct": 0.667},
    {"name": "Buddy Hield", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.332, "underPct": 0.668},
    {"name": "D'Angelo Russell", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.321, "underPct": 0.679},
    {"name": "Mark Williams", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.32, "underPct": 0.68},
    {"name": "Khris Middleton", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.315, "underPct": 0.685},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.311, "underPct": 0.689},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.311, "underPct": 0.689},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.294, "underPct": 0.706},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.268, "underPct": 0.732},
    {"name": "Cameron Johnson", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.255, "underPct": 0.745},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.253, "underPct": 0.747},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.236, "underPct": 0.764},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.219, "underPct": 0.781},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.21, "underPct": 0.79},
    {"name": "Santi Aldama", "line": 17.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.176, "underPct": 0.824},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.175, "underPct": 0.825},
    {"name": "Kobe Sanders", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.168, "underPct": 0.832},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.154, "underPct": 0.846},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.146, "underPct": 0.854},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.145, "underPct": 0.855},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.118, "underPct": 0.882},
    {"name": "Cole Anthony", "line": 10.5, "l5": 0.0, "l10": 0.4, "l15": 0.4, "overPct": 0.106, "underPct": 0.894},
    {"name": "Keldon Johnson", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.102, "underPct": 0.898},
    {"name": "Tyus Jones", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.044, "underPct": 0.956},
];const underdogAssistsHitRates = [
    {"name": "Paul George", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.832, "underPct": 0.168},
    {"name": "Jrue Holiday", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.739, "underPct": 0.261},
    {"name": "Alperen Sengun", "line": 6.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.614, "underPct": 0.386},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.612, "underPct": 0.388},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.607, "underPct": 0.393},
    {"name": "Cole Anthony", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.594, "underPct": 0.406},
    {"name": "Al Horford", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.593, "underPct": 0.407},
    {"name": "Terance Mann", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.562, "underPct": 0.438},
    {"name": "Luke Kennard", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.547, "underPct": 0.453},
    {"name": "Amen Thompson", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.547, "underPct": 0.453},
    {"name": "VJ Edgecombe", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.514, "underPct": 0.486},
    {"name": "Buddy Hield", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.51, "underPct": 0.49},
    {"name": "Zach LaVine", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.499, "underPct": 0.501},
    {"name": "Norman Powell", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.492, "underPct": 0.508},
    {"name": "Zion Williamson", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.485, "underPct": 0.515},
    {"name": "Davion Mitchell", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.481, "underPct": 0.519},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.442, "underPct": 0.558},
    {"name": "Max Christie", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.432, "underPct": 0.568},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Anthony Black", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Scottie Barnes", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.42, "underPct": 0.58},
    {"name": "Cedric Coward", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "D'Angelo Russell", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.393, "underPct": 0.607},
    {"name": "Draymond Green", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.383, "underPct": 0.617},
    {"name": "Devin Booker", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.365, "underPct": 0.635},
    {"name": "Kyle Kuzma", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.361, "underPct": 0.639},
    {"name": "Jeremiah Fears", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.337, "underPct": 0.663},
    {"name": "Naji Marshall", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.323, "underPct": 0.677},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.278, "underPct": 0.722},
    {"name": "Tristan da Silva", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.247, "underPct": 0.753},
    {"name": "Zach Edey", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.239, "underPct": 0.761},
];const underdogReboundsHitRates = [
    {"name": "Matas Buzelis", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.766, "underPct": 0.234},
    {"name": "Alperen Sengun", "line": 9.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.691, "underPct": 0.309},
    {"name": "Cedric Coward", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.685, "underPct": 0.315},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.669, "underPct": 0.331},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.655, "underPct": 0.345},
    {"name": "Brandon Williams", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.644, "underPct": 0.356},
    {"name": "Cooper Flagg", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.594, "underPct": 0.406},
    {"name": "Isaac Okoro", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.594, "underPct": 0.406},
    {"name": "Jrue Holiday", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.546, "underPct": 0.454},
    {"name": "P.J. Washington", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.529, "underPct": 0.471},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.516, "underPct": 0.484},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.514, "underPct": 0.486},
    {"name": "Naz Reid", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.499, "underPct": 0.501},
    {"name": "Payton Pritchard", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.48, "underPct": 0.52},
    {"name": "Derik Queen", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.46, "underPct": 0.54},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.456, "underPct": 0.544},
    {"name": "Al Horford", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.445, "underPct": 0.555},
    {"name": "Zach Edey", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.437, "underPct": 0.563},
    {"name": "Alex Sarr", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.432, "underPct": 0.568},
    {"name": "Donte DiVincenzo", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.423, "underPct": 0.577},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.422, "underPct": 0.578},
    {"name": "Shaedon Sharpe", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.417, "underPct": 0.583},
    {"name": "Toumani Camara", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.412, "underPct": 0.588},
    {"name": "Norman Powell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.39, "underPct": 0.61},
    {"name": "Anfernee Simons", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.386, "underPct": 0.614},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.374, "underPct": 0.626},
    {"name": "Moses Moody", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.37, "underPct": 0.63},
    {"name": "Luka Garza", "line": 5.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.365, "underPct": 0.635},
    {"name": "Yves Missi", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.363, "underPct": 0.637},
    {"name": "Ryan Dunn", "line": 4.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.361, "underPct": 0.639},
    {"name": "Kevin Durant", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.357, "underPct": 0.643},
    {"name": "Will Richard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.336, "underPct": 0.664},
    {"name": "Daniel Gafford", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.319, "underPct": 0.681},
    {"name": "John Collins", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.297, "underPct": 0.703},
    {"name": "Kelly Olynyk", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.242, "underPct": 0.758},
    {"name": "Mark Williams", "line": 9.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.232, "underPct": 0.768},
    {"name": "Kobe Sanders", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.229, "underPct": 0.771},
    {"name": "Brook Lopez", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.208, "underPct": 0.792},
    {"name": "Julian Champagnie", "line": 4.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.172, "underPct": 0.828},
];const underdogBlocksHitRates = [
    {"name": "Scottie Barnes", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.61, "underPct": 0.39},
    {"name": "Alex Sarr", "line": 1.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.652, "underPct": 0.348},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
];const underdogStealsHitRates = [
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.413, "underPct": 0.587},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.364, "underPct": 0.636},
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.526, "underPct": 0.474},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Kel'el Ware", "line": 21.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Zaccharie Risacher", "line": 15.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 38.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 38.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jrue Holiday", "line": 25.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Al Horford", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 32.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 33.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Simone Fontecchio", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaac Okoro", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 41.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dereck Lively II", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Grayson Allen", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Stephen Curry", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 37.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Suggs", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jock Landale", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Justin Edwards", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anfernee Simons", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 27.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Day'Ron Sharpe", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Dunn", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 36.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Khris Middleton", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandin Podziemski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 36.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Buddy Hield", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gradey Dick", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Hawkins", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 39.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dru Smith", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pelle Larsson", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Vassell", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Santi Aldama", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremy Sochan", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julian Champagnie", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paul George", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 40.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 28.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 20.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bobby Portis", "line": 24.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPRHitRates = [
    {"name": "Kel'el Ware", "line": 21.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Aaron Gordon", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 31.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Grayson Allen", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ivica Zubac", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Coby White", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bam Adebayo", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Santi Aldama", "line": 25.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "De'Aaron Fox", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Brown", "line": 33.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogPAHitRates = [
    {"name": "Shaedon Sharpe", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Grayson Allen", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jrue Holiday", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Franz Wagner", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Edwards", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zion Williamson", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 36.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const underdogRAHitRates = [
    {"name": "Kel'el Ware", "line": 10.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 16.5, "l5": 1.0, "l10": 1.0, "l15": 0.73, "overPct": 1.0, "underPct": 0.0},
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cole Anthony", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jrue Holiday", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Immanuel Quickley", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derrick White", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Coby White", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Santi Aldama", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremy Sochan", "line": 7.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kyle Kuzma", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 9.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const underdogTurnoversHitRates = [
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephen Curry", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
];const underdogBlocksStealsHitRates = [
    {"name": "Myles Turner", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Derik Queen", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Rudy Gobert", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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

