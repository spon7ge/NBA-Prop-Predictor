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
    {"name1": "VJ Edgecombe", "name2": "Derik Queen", "line1": 12.5, "line2": 13.5, "prediction1": 20.77, "prediction2": 22.8, "side1": "over", "side2": "over", "recommendation": 1, "ev": 136.42, "kelly": 0.682, "sigma1": "Med", "sigma2": "High", "prob1": 0.904, "prob2": 0.89, "hitRate1": 48.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 59.8, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Jeremy Sochan", "name2": "Draymond Green", "line1": 5.5, "line2": 9.5, "prediction1": 11.75, "prediction2": 17.37, "side1": "over", "side2": "over", "recommendation": 1, "ev": 122.76, "kelly": 0.614, "sigma1": "Med", "sigma2": "High", "prob1": 0.875, "prob2": 0.866, "hitRate1": 55.7, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 26.3, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "Jaden McDaniels", "name2": "Jaylin Williams", "line1": 13.5, "line2": 4.5, "prediction1": 20.19, "prediction2": 8.97, "side1": "over", "side2": "over", "recommendation": 1, "ev": 106.7, "kelly": 0.533, "sigma1": "High", "sigma2": "Low", "prob1": 0.844, "prob2": 0.833, "hitRate1": 31.6, "l5_1": 0.0, "l15_1": 0.53, "hitRate2": 80.6, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Paul George", "name2": "Donte DiVincenzo", "line1": 15.5, "line2": 13.5, "prediction1": 21.84, "prediction2": 19.3, "side1": "over", "side2": "over", "recommendation": 1, "ev": 102.65, "kelly": 0.513, "sigma1": "High", "sigma2": "Med", "prob1": 0.832, "prob2": 0.828, "hitRate1": 30.9, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 47.7, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Mike Conley", "name2": "Jalen Williams", "line1": 4.5, "line2": 18.5, "prediction1": 8.28, "prediction2": 24.34, "side1": "over", "side2": "over", "recommendation": 0, "ev": 93.84, "kelly": 0.469, "sigma1": "Low", "sigma2": "Med", "prob1": 0.805, "prob2": 0.819, "hitRate1": 55.2, "l5_1": 0.4, "l15_1": 0.8, "hitRate2": 25.6, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Jordan Walsh", "name2": "Cason Wallace", "line1": 5.5, "line2": 7.5, "prediction1": 9.8, "prediction2": 12.89, "side1": "over", "side2": "over", "recommendation": 1, "ev": 88.08, "kelly": 0.44, "sigma1": "Low", "sigma2": "High", "prob1": 0.8, "prob2": 0.799, "hitRate1": 59.0, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 64.9, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Kyshawn George", "name2": "Devin Vassell", "line1": 15.0, "line2": 17.5, "prediction1": 21.72, "prediction2": 24.34, "side1": "over", "side2": "over", "recommendation": 1, "ev": 83.33, "kelly": 0.417, "sigma1": "High", "sigma2": "High", "prob1": 0.785, "prob2": 0.794, "hitRate1": 37.8, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 61.5, "l5_2": 0.6, "l15_2": 0.2},
    {"name1": "Saddiq Bey", "name2": "Quinten Post", "line1": 15.5, "line2": 8.0, "prediction1": 21.39, "prediction2": 12.54, "side1": "over", "side2": "over", "recommendation": 1, "ev": 71.6, "kelly": 0.358, "sigma1": "High", "sigma2": "Med", "prob1": 0.763, "prob2": 0.765, "hitRate1": 42.8, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 47.4, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Luke Kornet", "name2": "Jordan Hawkins", "line1": 7.5, "line2": 5.5, "prediction1": 11.38, "prediction2": 8.79, "side1": "over", "side2": "over", "recommendation": 0, "ev": 70.25, "kelly": 0.351, "sigma1": "Low", "sigma2": "Low", "prob1": 0.762, "prob2": 0.76, "hitRate1": 44.6, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 61.0, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Jared McCain", "name2": "Naz Reid", "line1": 14.5, "line2": 13.5, "prediction1": 11.76, "prediction2": 18.13, "side1": "under", "side2": "over", "recommendation": 0, "ev": 68.19, "kelly": 0.341, "sigma1": "Low", "sigma2": "Med", "prob1": 0.757, "prob2": 0.755, "hitRate1": 95.2, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 59.1, "l5_2": 0.4, "l15_2": 0.47},
];const prizepicksTriosData = [
    {"name1": "VJ Edgecombe", "name2": "Jeremy Sochan", "name3": "Derik Queen", "line1": 12.5, "line2": 5.5, "line3": 13.5, "prediction1": 20.77, "prediction2": 11.75, "prediction3": 22.8, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 280.15, "kelly": 0.56, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.904, "prob2": 0.875, "prob3": 0.89, "hitRate1": 48.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 55.7, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 59.8, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Paul George", "name2": "Jaden McDaniels", "name3": "Draymond Green", "line1": 15.5, "line2": 13.5, "line3": 9.5, "prediction1": 21.84, "prediction2": 20.19, "prediction3": 17.37, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 228.33, "kelly": 0.457, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.832, "prob2": 0.844, "prob3": 0.866, "hitRate1": 30.9, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 31.6, "l5_2": 0.0, "l15_2": 0.53, "hitRate3": 26.3, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Donte DiVincenzo", "name2": "Jordan Walsh", "name3": "Jalen Williams", "line1": 13.5, "line2": 5.5, "line3": 18.5, "prediction1": 19.3, "prediction2": 9.8, "prediction3": 24.34, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 193.25, "kelly": 0.386, "sigma1": "Med", "sigma2": "Low", "sigma3": "Med", "prob1": 0.828, "prob2": 0.8, "prob3": 0.819, "hitRate1": 47.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 59.0, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 25.6, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Kyshawn George", "name2": "Devin Vassell", "name3": "Cason Wallace", "line1": 15.0, "line2": 17.5, "line3": 7.5, "prediction1": 21.72, "prediction2": 24.34, "prediction3": 12.89, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 169.15, "kelly": 0.338, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.785, "prob2": 0.794, "prob3": 0.799, "hitRate1": 37.8, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 61.5, "l5_2": 0.6, "l15_2": 0.2, "hitRate3": 64.9, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Luke Kornet", "name2": "Saddiq Bey", "name3": "Quinten Post", "line1": 7.5, "line2": 15.5, "line3": 8.0, "prediction1": 11.38, "prediction2": 21.39, "prediction3": 12.54, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 140.1, "kelly": 0.28, "sigma1": "Low", "sigma2": "High", "sigma3": "Med", "prob1": 0.762, "prob2": 0.763, "prob3": 0.765, "hitRate1": 44.6, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 42.8, "l5_2": 1.0, "l15_2": 0.53, "hitRate3": 47.4, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Jared McCain", "name2": "Jordan Hawkins", "name3": "Neemias Queta", "line1": 14.5, "line2": 5.5, "line3": 9.5, "prediction1": 11.76, "prediction2": 8.79, "prediction3": 13.82, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 132.41, "kelly": 0.265, "sigma1": "Low", "sigma2": "Low", "sigma3": "Med", "prob1": 0.757, "prob2": 0.76, "prob3": 0.748, "hitRate1": 95.2, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 61.0, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 69.9, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Marvin Bagley III", "name2": "Naz Reid", "name3": "Will Richard", "line1": 12.5, "line2": 13.5, "line3": 6.5, "prediction1": 10.88, "prediction2": 18.13, "prediction3": 9.41, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 116.28, "kelly": 0.233, "sigma1": "Low", "sigma2": "Med", "sigma3": "Low", "prob1": 0.734, "prob2": 0.755, "prob3": 0.722, "hitRate1": 84.6, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 59.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 65.4, "l5_3": 0.4, "l15_3": 0.6},
    {"name1": "Gradey Dick", "name2": "Kentavious Caldwell-Pope", "name3": "Trey Murphy III", "line1": 6.5, "line2": 7.5, "line3": 20.5, "prediction1": 9.28, "prediction2": 10.71, "prediction3": 16.18, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 101.56, "kelly": 0.203, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "prob1": 0.711, "prob2": 0.715, "prob3": 0.735, "hitRate1": 22.9, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 34.0, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 65.7, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Andre Drummond", "name2": "Julian Champagnie", "name3": "Anfernee Simons", "line1": 8.5, "line2": 11.5, "line3": 11.5, "prediction1": 11.08, "prediction2": 15.41, "prediction3": 15.28, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 82.02, "kelly": 0.164, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.702, "prob2": 0.69, "prob3": 0.696, "hitRate1": 33.9, "l5_1": 0.2, "l15_1": 0.47, "hitRate2": 62.6, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 51.0, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Quentin Grimes", "name2": "Harrison Barnes", "name3": "Isaiah Joe", "line1": 17.5, "line2": 12.5, "line3": 9.5, "prediction1": 15.14, "prediction2": 16.25, "prediction3": 12.48, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 72.89, "kelly": 0.146, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.698, "prob2": 0.678, "prob3": 0.676, "hitRate1": 42.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 69.8, "l5_2": 0.4, "l15_2": 0.53, "hitRate3": 70.3, "l5_3": 0.8, "l15_3": 0.73},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "VJ Edgecombe", "name2": "Derik Queen", "line1": 12.5, "line2": 13.5, "prediction1": 20.77, "prediction2": 22.8, "side1": "over", "side2": "over", "recommendation": 1, "ev": 136.42, "kelly": 0.682, "sigma1": "Med", "sigma2": "High", "prob1": 0.904, "prob2": 0.89, "hitRate1": 55.7, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 59.8, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Tyrese Maxey", "name2": "Jaden McDaniels", "line1": 32.5, "line2": 13.5, "prediction1": 26.68, "prediction2": 20.19, "side1": "under", "side2": "over", "recommendation": 1, "ev": 120.63, "kelly": 0.603, "sigma1": "Med", "sigma2": "High", "prob1": 0.889, "prob2": 0.844, "hitRate1": 37.7, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 31.6, "l5_2": 0.0, "l15_2": 0.53},
    {"name1": "Paul George", "name2": "Jaylin Williams", "line1": 15.5, "line2": 4.5, "prediction1": 21.84, "prediction2": 8.97, "side1": "over", "side2": "over", "recommendation": 1, "ev": 103.83, "kelly": 0.519, "sigma1": "High", "sigma2": "Low", "prob1": 0.832, "prob2": 0.833, "hitRate1": 34.2, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 80.6, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Donte DiVincenzo", "name2": "Jalen Williams", "line1": 13.5, "line2": 18.5, "prediction1": 19.3, "prediction2": 24.34, "side1": "over", "side2": "over", "recommendation": 1, "ev": 99.49, "kelly": 0.497, "sigma1": "Med", "sigma2": "Med", "prob1": 0.828, "prob2": 0.819, "hitRate1": 47.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 25.6, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Kyshawn George", "name2": "Cason Wallace", "line1": 15.5, "line2": 7.5, "prediction1": 21.72, "prediction2": 12.89, "side1": "over", "side2": "over", "recommendation": 1, "ev": 80.01, "kelly": 0.4, "sigma1": "High", "sigma2": "High", "prob1": 0.766, "prob2": 0.799, "hitRate1": 46.0, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 64.9, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Quentin Grimes", "name2": "Saddiq Bey", "line1": 17.5, "line2": 15.5, "prediction1": 15.14, "prediction2": 21.39, "side1": "under", "side2": "over", "recommendation": 0, "ev": 56.56, "kelly": 0.283, "sigma1": "Med", "sigma2": "High", "prob1": 0.698, "prob2": 0.763, "hitRate1": 42.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 42.8, "l5_2": 1.0, "l15_2": 0.53},
    {"name1": "Julian Champagnie", "name2": "Naz Reid", "line1": 11.5, "line2": 13.5, "prediction1": 15.41, "prediction2": 18.13, "side1": "over", "side2": "over", "recommendation": 0, "ev": 53.28, "kelly": 0.266, "sigma1": "High", "sigma2": "Med", "prob1": 0.69, "prob2": 0.755, "hitRate1": 62.6, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 59.1, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Trey Murphy III", "name2": "Jaylen Brown", "line1": 19.5, "line2": 27.5, "prediction1": 16.18, "prediction2": 31.24, "side1": "under", "side2": "over", "recommendation": 0, "ev": 38.04, "kelly": 0.19, "sigma1": "High", "sigma2": "High", "prob1": 0.689, "prob2": 0.681, "hitRate1": 57.1, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 68.8, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Donovan Clingan", "name2": "Harrison Barnes", "line1": 8.5, "line2": 12.5, "prediction1": 11.32, "prediction2": 16.25, "side1": "over", "side2": "over", "recommendation": 0, "ev": 35.65, "kelly": 0.178, "sigma1": "Med", "sigma2": "High", "prob1": 0.68, "prob2": 0.678, "hitRate1": 88.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 69.8, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "Cedric Coward", "name2": "Brandin Podziemski", "line1": 13.5, "line2": 13.5, "prediction1": 17.15, "prediction2": 17.15, "side1": "over", "side2": "over", "recommendation": 0, "ev": 31.73, "kelly": 0.159, "sigma1": "High", "sigma2": "High", "prob1": 0.67, "prob2": 0.669, "hitRate1": 38.4, "l5_1": 0.0, "l15_1": 0.27, "hitRate2": 45.3, "l5_2": 0.8, "l15_2": 0.53},
];const underdogTriosData = [
    {"name1": "VJ Edgecombe", "name2": "Derik Queen", "name3": "Jaylin Williams", "line1": 12.5, "line2": 13.5, "line3": 4.5, "prediction1": 20.77, "prediction2": 22.8, "prediction3": 8.97, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 261.72, "kelly": 0.523, "sigma1": "Med", "sigma2": "High", "sigma3": "Low", "prob1": 0.904, "prob2": 0.89, "prob3": 0.833, "hitRate1": 55.7, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 59.8, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 80.6, "l5_3": 0.8, "l15_3": 0.6},
    {"name1": "Tyrese Maxey", "name2": "Jaden McDaniels", "name3": "Jalen Williams", "line1": 32.5, "line2": 13.5, "line3": 18.5, "prediction1": 26.68, "prediction2": 20.19, "prediction3": 24.34, "side1": "under", "side2": "over", "side3": "over", "recommendation": 1, "ev": 232.02, "kelly": 0.464, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.889, "prob2": 0.844, "prob3": 0.819, "hitRate1": 37.7, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 31.6, "l5_2": 0.0, "l15_2": 0.53, "hitRate3": 25.6, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Paul George", "name2": "Donte DiVincenzo", "name3": "Cason Wallace", "line1": 15.5, "line2": 13.5, "line3": 7.5, "prediction1": 21.84, "prediction2": 19.3, "prediction3": 12.89, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 197.51, "kelly": 0.395, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.832, "prob2": 0.828, "prob3": 0.799, "hitRate1": 34.2, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 47.7, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 64.9, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Kyshawn George", "name2": "Julian Champagnie", "name3": "Saddiq Bey", "line1": 15.5, "line2": 11.5, "line3": 15.5, "prediction1": 21.72, "prediction2": 15.41, "prediction3": 21.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 117.77, "kelly": 0.236, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.766, "prob2": 0.69, "prob3": 0.763, "hitRate1": 46.0, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 62.6, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 42.8, "l5_3": 1.0, "l15_3": 0.53},
    {"name1": "Quentin Grimes", "name2": "Naz Reid", "name3": "Jaylen Brown", "line1": 17.5, "line2": 13.5, "line3": 27.5, "prediction1": 15.14, "prediction2": 18.13, "prediction3": 31.24, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 94.04, "kelly": 0.188, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.698, "prob2": 0.755, "prob3": 0.681, "hitRate1": 42.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 59.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 68.8, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Donovan Clingan", "name2": "Harrison Barnes", "name3": "Trey Murphy III", "line1": 8.5, "line2": 12.5, "line3": 19.5, "prediction1": 11.32, "prediction2": 16.25, "prediction3": 16.18, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 71.69, "kelly": 0.143, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.68, "prob2": 0.678, "prob3": 0.689, "hitRate1": 88.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 69.8, "l5_2": 0.4, "l15_2": 0.53, "hitRate3": 57.1, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Cedric Coward", "name2": "Jalen Brunson", "name3": "Brandin Podziemski", "line1": 13.5, "line2": 27.5, "line3": 13.5, "prediction1": 17.15, "prediction2": 25.17, "prediction3": 17.15, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 58.15, "kelly": 0.116, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.67, "prob2": 0.654, "prob3": 0.669, "hitRate1": 38.4, "l5_1": 0.0, "l15_1": 0.27, "hitRate2": 30.1, "l5_2": 0.6, "l15_2": 0.67, "hitRate3": 45.3, "l5_3": 0.8, "l15_3": 0.53},
    {"name1": "Jamal Shead", "name2": "Keldon Johnson", "name3": "Aaron Wiggins", "line1": 6.5, "line2": 12.5, "line3": 11.5, "prediction1": 8.46, "prediction2": 15.45, "prediction3": 14.26, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 44.29, "kelly": 0.089, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.653, "prob2": 0.641, "prob3": 0.639, "hitRate1": 25.1, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 81.6, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 67.5, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "Cam Spencer", "name2": "Julius Randle", "name3": "Josh Hart", "line1": 10.5, "line2": 21.5, "line3": 12.5, "prediction1": 13.0, "prediction2": 23.66, "prediction3": 15.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 33.27, "kelly": 0.067, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.64, "prob2": 0.604, "prob3": 0.638, "hitRate1": 70.5, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 27.0, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 70.7, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Jaylen Wells", "name2": "Karl-Anthony Towns", "name3": "Moses Moody", "line1": 12.5, "line2": 22.5, "line3": 12.5, "prediction1": 14.55, "prediction2": 20.76, "prediction3": 14.56, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 18.88, "kelly": 0.038, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.599, "prob2": 0.611, "prob3": 0.601, "hitRate1": 64.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 48.4, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 55.5, "l5_3": 0.2, "l15_3": 0.4},
];const prizepicksPointsHitRates = [
    {"name": "Shaedon Sharpe", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.89, "underPct": 0.11},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.82, "underPct": 0.18},
    {"name": "Keldon Johnson", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.816, "underPct": 0.184},
    {"name": "Donovan Clingan", "line": 9.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.814, "underPct": 0.186},
    {"name": "Jaylin Williams", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.806, "underPct": 0.194},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.707, "underPct": 0.293},
    {"name": "Cam Spencer", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.705, "underPct": 0.295},
    {"name": "Isaiah Joe", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.703, "underPct": 0.297},
    {"name": "Jalen Brunson", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.699, "underPct": 0.301},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.699, "underPct": 0.301},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.698, "underPct": 0.302},
    {"name": "Jaylen Brown", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.688, "underPct": 0.312},
    {"name": "Shai Gilgeous-Alexander", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.687, "underPct": 0.313},
    {"name": "Mikal Bridges", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.686, "underPct": 0.314},
    {"name": "Miles McBride", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.678, "underPct": 0.322},
    {"name": "Aaron Wiggins", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.675, "underPct": 0.325},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.66, "underPct": 0.34},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.654, "underPct": 0.346},
    {"name": "Will Richard", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.654, "underPct": 0.346},
    {"name": "Cason Wallace", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.649, "underPct": 0.351},
    {"name": "Jakob Poeltl", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.647, "underPct": 0.353},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.643, "underPct": 0.357},
    {"name": "Chet Holmgren", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.64, "underPct": 0.36},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.626, "underPct": 0.374},
    {"name": "Tyrese Maxey", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.623, "underPct": 0.377},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.622, "underPct": 0.378},
    {"name": "Devin Vassell", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.615, "underPct": 0.385},
    {"name": "Jordan Hawkins", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.61, "underPct": 0.39},
    {"name": "Santi Aldama", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.607, "underPct": 0.393},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.598, "underPct": 0.402},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.591, "underPct": 0.409},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.59, "underPct": 0.41},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.575, "underPct": 0.425},
    {"name": "Deni Avdija", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.557, "underPct": 0.443},
    {"name": "Jeremy Sochan", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.557, "underPct": 0.443},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.555, "underPct": 0.445},
    {"name": "Mike Conley", "line": 4.5, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jeremiah Fears", "line": 15.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.539, "underPct": 0.461},
    {"name": "Ajay Mitchell", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.524, "underPct": 0.476},
    {"name": "Dylan Harper", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.523, "underPct": 0.477},
    {"name": "Karl-Anthony Towns", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.516, "underPct": 0.484},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.514, "underPct": 0.486},
    {"name": "Anfernee Simons", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.51, "underPct": 0.49},
    {"name": "Scottie Barnes", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.493, "underPct": 0.507},
    {"name": "Derrick White", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.49, "underPct": 0.51},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.489, "underPct": 0.511},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.481, "underPct": 0.519},
    {"name": "VJ Edgecombe", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.53, "overPct": 0.481, "underPct": 0.519},
    {"name": "Micah Peavy", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.479, "underPct": 0.521},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.477, "underPct": 0.523},
    {"name": "Quinten Post", "line": 8.0, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.474, "underPct": 0.526},
    {"name": "Brandin Podziemski", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.453, "underPct": 0.547},
    {"name": "Luke Kornet", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.446, "underPct": 0.554},
    {"name": "Jordan Clarkson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.435, "underPct": 0.565},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.429, "underPct": 0.571},
    {"name": "Saddiq Bey", "line": 15.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.428, "underPct": 0.572},
    {"name": "Mitchell Robinson", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.392, "underPct": 0.608},
    {"name": "Ja'Kobe Walter", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.391, "underPct": 0.609},
    {"name": "Sam Hauser", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.39, "underPct": 0.61},
    {"name": "Buddy Hield", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.389, "underPct": 0.611},
    {"name": "Cedric Coward", "line": 13.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.384, "underPct": 0.616},
    {"name": "Kyshawn George", "line": 15.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.378, "underPct": 0.622},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.367, "underPct": 0.633},
    {"name": "Gary Payton II", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.348, "underPct": 0.652},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.343, "underPct": 0.657},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.34, "underPct": 0.66},
    {"name": "Andre Drummond", "line": 8.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.339, "underPct": 0.661},
    {"name": "Immanuel Quickley", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.338, "underPct": 0.662},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.316, "underPct": 0.684},
    {"name": "Paul George", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.309, "underPct": 0.691},
    {"name": "Yves Missi", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.279, "underPct": 0.721},
    {"name": "Julius Randle", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.27, "underPct": 0.73},
    {"name": "Draymond Green", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.263, "underPct": 0.737},
    {"name": "Jalen Williams", "line": 18.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.256, "underPct": 0.744},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.251, "underPct": 0.749},
    {"name": "Gradey Dick", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.229, "underPct": 0.771},
    {"name": "Jonathan Kuminga", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.216, "underPct": 0.784},
    {"name": "Cam Whitmore", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.192, "underPct": 0.808},
    {"name": "Bilal Coulibaly", "line": 11.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.169, "underPct": 0.831},
    {"name": "Marvin Bagley III", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.154, "underPct": 0.846},
    {"name": "Jared McCain", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.048, "underPct": 0.952},
];const prizepicksAssistsHitRates = [
    {"name": "Rudy Gobert", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.752, "underPct": 0.248},
    {"name": "Karl-Anthony Towns", "line": 2.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.662, "underPct": 0.338},
    {"name": "Jaylen Wells", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.638, "underPct": 0.362},
    {"name": "Draymond Green", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.579, "underPct": 0.421},
    {"name": "Josh Hart", "line": 5.0, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.576, "underPct": 0.424},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.572, "underPct": 0.428},
    {"name": "De'Aaron Fox", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.558, "underPct": 0.442},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.542, "underPct": 0.458},
    {"name": "Kentavious Caldwell-Pope", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.538, "underPct": 0.462},
    {"name": "Anfernee Simons", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.534, "underPct": 0.466},
    {"name": "Shai Gilgeous-Alexander", "line": 6.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.511, "underPct": 0.489},
    {"name": "Jaylen Brown", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.505, "underPct": 0.495},
    {"name": "Neemias Queta", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.503, "underPct": 0.497},
    {"name": "Kyshawn George", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.483, "underPct": 0.517},
    {"name": "Jamal Shead", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.483, "underPct": 0.517},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.47, "underPct": 0.53},
    {"name": "Payton Pritchard", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.469, "underPct": 0.531},
    {"name": "Julian Champagnie", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.453, "underPct": 0.547},
    {"name": "Toumani Camara", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.452, "underPct": 0.548},
    {"name": "Deni Avdija", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.448, "underPct": 0.552},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Jose Alvarado", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.422, "underPct": 0.578},
    {"name": "Cam Spencer", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.383, "underPct": 0.617},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.373, "underPct": 0.627},
    {"name": "Derrick White", "line": 5.0, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.366, "underPct": 0.634},
    {"name": "Andre Drummond", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.364, "underPct": 0.636},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.346, "underPct": 0.654},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.34, "underPct": 0.66},
    {"name": "Kris Murray", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.334, "underPct": 0.666},
    {"name": "Tyrese Maxey", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.324, "underPct": 0.676},
    {"name": "Jalen Williams", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.308, "underPct": 0.692},
];const prizepicksReboundsHitRates = [
    {"name": "Donovan Clingan", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.68, "underPct": 0.32},
    {"name": "Keldon Johnson", "line": 6.0, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.663, "underPct": 0.337},
    {"name": "Saddiq Bey", "line": 6.0, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.658, "underPct": 0.342},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Jerami Grant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.622, "underPct": 0.378},
    {"name": "Andre Drummond", "line": 10.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.621, "underPct": 0.379},
    {"name": "Jaylen Brown", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.608, "underPct": 0.392},
    {"name": "Shaedon Sharpe", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.607, "underPct": 0.393},
    {"name": "Deni Avdija", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.593, "underPct": 0.407},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.569, "underPct": 0.431},
    {"name": "Cedric Coward", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.564, "underPct": 0.436},
    {"name": "Mitchell Robinson", "line": 7.0, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.562, "underPct": 0.438},
    {"name": "Jaden McDaniels", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.555, "underPct": 0.445},
    {"name": "Draymond Green", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.548, "underPct": 0.452},
    {"name": "Paul George", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.529, "underPct": 0.471},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.522, "underPct": 0.478},
    {"name": "Naz Reid", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.521, "underPct": 0.479},
    {"name": "Yves Missi", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.515, "underPct": 0.485},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.513, "underPct": 0.487},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.51, "underPct": 0.49},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.507, "underPct": 0.493},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Karl-Anthony Towns", "line": 11.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.501, "underPct": 0.499},
    {"name": "Josh Hart", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.492, "underPct": 0.508},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.477, "underPct": 0.523},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.474, "underPct": 0.526},
    {"name": "Derrick White", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.47, "underPct": 0.53},
    {"name": "Sandro Mamukelashvili", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.465, "underPct": 0.535},
    {"name": "Jakob Poeltl", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.463, "underPct": 0.537},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.456, "underPct": 0.544},
    {"name": "Quinten Post", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.444, "underPct": 0.556},
    {"name": "Jalen Williams", "line": 5.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.441, "underPct": 0.559},
    {"name": "Jonathan Kuminga", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.436, "underPct": 0.564},
    {"name": "Immanuel Quickley", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.431, "underPct": 0.569},
    {"name": "Jamal Shead", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.418, "underPct": 0.582},
    {"name": "Scottie Barnes", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.405, "underPct": 0.595},
    {"name": "Luke Kornet", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.398, "underPct": 0.602},
    {"name": "Trey Murphy III", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.393, "underPct": 0.607},
    {"name": "Zach Edey", "line": 12.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.39, "underPct": 0.61},
    {"name": "Jordan Walsh", "line": 5.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.378, "underPct": 0.622},
    {"name": "Santi Aldama", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.376, "underPct": 0.624},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.375, "underPct": 0.625},
    {"name": "Harrison Barnes", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.373, "underPct": 0.627},
    {"name": "Isaiah Joe", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.367, "underPct": 0.633},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.359, "underPct": 0.641},
    {"name": "Chet Holmgren", "line": 9.0, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.322, "underPct": 0.678},
    {"name": "Shai Gilgeous-Alexander", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.322, "underPct": 0.678},
    {"name": "Micah Peavy", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.32, "underPct": 0.68},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.32, "underPct": 0.68},
    {"name": "Derik Queen", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.278, "underPct": 0.722},
    {"name": "Devin Vassell", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.277, "underPct": 0.723},
    {"name": "Brandin Podziemski", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.27, "underPct": 0.73},
    {"name": "Bilal Coulibaly", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.269, "underPct": 0.731},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.246, "underPct": 0.754},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.21, "underPct": 0.79},
    {"name": "Marvin Bagley III", "line": 7.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.14, "underPct": 0.86},
    {"name": "Aaron Wiggins", "line": 4.0, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.139, "underPct": 0.861},
];const prizepicksBlocksHitRates = [
    {"name": "Bilal Coulibaly", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.566, "underPct": 0.434},
    {"name": "Kyshawn George", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.465, "underPct": 0.535},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.505, "underPct": 0.495},
    {"name": "Toumani Camara", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.469, "underPct": 0.531},
    {"name": "Anthony Edwards", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.493, "underPct": 0.507},
    {"name": "Moses Moody", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.502, "underPct": 0.498},
    {"name": "Quinten Post", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
];const prizepicksStealsHitRates = [
    {"name": "Paul George", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.587, "underPct": 0.413},
    {"name": "Cam Whitmore", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.459, "underPct": 0.541},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.587, "underPct": 0.413},
    {"name": "Luke Kornet", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.262, "underPct": 0.738},
    {"name": "Santi Aldama", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.553, "underPct": 0.447},
    {"name": "Cam Spencer", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.479, "underPct": 0.521},
    {"name": "Zach Edey", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.524, "underPct": 0.476},
    {"name": "Rudy Gobert", "line": 0.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.294, "underPct": 0.706},
    {"name": "Karl-Anthony Towns", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.555, "underPct": 0.445},
    {"name": "Mitchell Robinson", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.577, "underPct": 0.423},
    {"name": "Cason Wallace", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.579, "underPct": 0.421},
    {"name": "Shai Gilgeous-Alexander", "line": 1.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.606, "underPct": 0.394},
    {"name": "Buddy Hield", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.64, "underPct": 0.36},
    {"name": "Gary Payton II", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.523, "underPct": 0.477},
    {"name": "Will Richard", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.598, "underPct": 0.402},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Saddiq Bey", "line": 24.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 39.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Wells", "line": 17.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 21.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Draymond Green", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Spencer", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 19.0, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylen Brown", "line": 38.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 21.0, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 39.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylin Williams", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luke Kornet", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gary Payton II", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mike Conley", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles McBride", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Joe", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quinten Post", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach Edey", "line": 28.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ajay Mitchell", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sam Hauser", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 41.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shaedon Sharpe", "line": 23.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 45.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Clarkson", "line": 15.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 26.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Buddy Hield", "line": 13.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Yves Missi", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 36.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paul George", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Marvin Bagley III", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Whitmore", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ja'Kobe Walter", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jakob Poeltl", "line": 22.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rudy Gobert", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jonathan Kuminga", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 21.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jared McCain", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Aaron Wiggins", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 34.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 35.0, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dylan Harper", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 28.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 19.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Williams", "line": 29.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPRHitRates = [
    {"name": "Keldon Johnson", "line": 19.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylin Williams", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Micah Peavy", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sandro Mamukelashvili", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Draymond Green", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Wells", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Santi Aldama", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Neemias Queta", "line": 19.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 15.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 22.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach Edey", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sam Hauser", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gary Payton II", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Buddy Hield", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quinten Post", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Joe", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 21.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ja'Kobe Walter", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Wiggins", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 21.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Whitmore", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Harrison Barnes", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kelly Olynyk", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 20.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Williams", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dylan Harper", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jared McCain", "line": 17.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Andre Drummond", "line": 20.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaden McDaniels", "line": 18.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jonathan Kuminga", "line": 16.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 15.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPAHitRates = [
    {"name": "Anthony Edwards", "line": 33.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 18.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylin Williams", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keldon Johnson", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Wells", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Spencer", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Santi Aldama", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Quinten Post", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Joe", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandin Podziemski", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mike Conley", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gary Payton II", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 18.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Wiggins", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Neemias Queta", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sam Hauser", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles McBride", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "VJ Edgecombe", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shaedon Sharpe", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Murray", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Marvin Bagley III", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Whitmore", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jared McCain", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Paul George", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Immanuel Quickley", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Harrison Barnes", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ja'Kobe Walter", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dylan Harper", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 21.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Naz Reid", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 9.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 22.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jonathan Kuminga", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Williams", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jaden McDaniels", "line": 16.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 14.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
];const prizepicksRAHitRates = [
    {"name": "Jaylen Brown", "line": 11.0, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Draymond Green", "line": 11.0, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 14.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 7.0, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 13.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Yves Missi", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 12.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 7.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anfernee Simons", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Hauser", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylin Williams", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quinten Post", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Wells", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ja'Kobe Walter", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 6.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sandro Mamukelashvili", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 12.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 13.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles McBride", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ajay Mitchell", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 14.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mike Conley", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaden McDaniels", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paul George", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donte DiVincenzo", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 14.0, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Naz Reid", "line": 8.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jared McCain", "line": 5.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksTurnoversHitRates = [
    {"name": "Julius Randle", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jose Alvarado", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Kyshawn George", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ja'Kobe Walter", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Yves Missi", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Marvin Bagley III", "line": 1.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Quentin Grimes", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sam Hauser", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jonathan Kuminga", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mike Conley", "line": 0.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const underdogPointsHitRates = [
    {"name": "Shaedon Sharpe", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.89, "underPct": 0.11},
    {"name": "Donovan Clingan", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.886, "underPct": 0.114},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.82, "underPct": 0.18},
    {"name": "Keldon Johnson", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.816, "underPct": 0.184},
    {"name": "Jaylin Williams", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.806, "underPct": 0.194},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.707, "underPct": 0.293},
    {"name": "Cam Spencer", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.705, "underPct": 0.295},
    {"name": "Jalen Brunson", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.699, "underPct": 0.301},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.698, "underPct": 0.302},
    {"name": "Jaylen Brown", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.688, "underPct": 0.312},
    {"name": "Miles McBride", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.678, "underPct": 0.322},
    {"name": "Aaron Wiggins", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.675, "underPct": 0.325},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.66, "underPct": 0.34},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.654, "underPct": 0.346},
    {"name": "Cason Wallace", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.649, "underPct": 0.351},
    {"name": "Jakob Poeltl", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.647, "underPct": 0.353},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.643, "underPct": 0.357},
    {"name": "Chet Holmgren", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.64, "underPct": 0.36},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.626, "underPct": 0.374},
    {"name": "Tyrese Maxey", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.623, "underPct": 0.377},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.621, "underPct": 0.379},
    {"name": "Santi Aldama", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.607, "underPct": 0.393},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.598, "underPct": 0.402},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.591, "underPct": 0.409},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.575, "underPct": 0.425},
    {"name": "Deni Avdija", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.557, "underPct": 0.443},
    {"name": "VJ Edgecombe", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.53, "overPct": 0.557, "underPct": 0.443},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.555, "underPct": 0.445},
    {"name": "Dylan Harper", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.523, "underPct": 0.477},
    {"name": "Karl-Anthony Towns", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.516, "underPct": 0.484},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.514, "underPct": 0.486},
    {"name": "Scottie Barnes", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.493, "underPct": 0.507},
    {"name": "Derrick White", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.49, "underPct": 0.51},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.481, "underPct": 0.519},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.477, "underPct": 0.523},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.46, "underPct": 0.54},
    {"name": "Brandin Podziemski", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.453, "underPct": 0.547},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.44, "underPct": 0.56},
    {"name": "Jordan Clarkson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.435, "underPct": 0.565},
    {"name": "Trey Murphy III", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.429, "underPct": 0.571},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.429, "underPct": 0.571},
    {"name": "Saddiq Bey", "line": 15.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.428, "underPct": 0.572},
    {"name": "Cedric Coward", "line": 13.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.384, "underPct": 0.616},
    {"name": "Paul George", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.342, "underPct": 0.658},
    {"name": "Immanuel Quickley", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.338, "underPct": 0.662},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.316, "underPct": 0.684},
    {"name": "Yves Missi", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.279, "underPct": 0.721},
    {"name": "Julius Randle", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.27, "underPct": 0.73},
    {"name": "Jalen Williams", "line": 18.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.256, "underPct": 0.744},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.251, "underPct": 0.749},
    {"name": "Cam Whitmore", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.192, "underPct": 0.808},
    {"name": "Bilal Coulibaly", "line": 11.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.169, "underPct": 0.831},
];const underdogAssistsHitRates = [
    {"name": "Jaylen Wells", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.638, "underPct": 0.362},
    {"name": "Jaylin Williams", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.633, "underPct": 0.367},
    {"name": "Kentavious Caldwell-Pope", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.538, "underPct": 0.462},
    {"name": "Neemias Queta", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.503, "underPct": 0.497},
    {"name": "Payton Pritchard", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.469, "underPct": 0.531},
    {"name": "Kris Murray", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.334, "underPct": 0.666},
];const underdogReboundsHitRates = [
    {"name": "Will Richard", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.686, "underPct": 0.314},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Jerami Grant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.622, "underPct": 0.378},
    {"name": "Cedric Coward", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.564, "underPct": 0.436},
    {"name": "Jaden McDaniels", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.555, "underPct": 0.445},
    {"name": "Paul George", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.55, "underPct": 0.45},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.549, "underPct": 0.451},
    {"name": "Mike Conley", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.521, "underPct": 0.479},
    {"name": "Naz Reid", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.521, "underPct": 0.479},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.51, "underPct": 0.49},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.456, "underPct": 0.544},
    {"name": "Quinten Post", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.444, "underPct": 0.556},
    {"name": "Jalen Williams", "line": 5.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.441, "underPct": 0.559},
    {"name": "Jamal Shead", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.418, "underPct": 0.582},
    {"name": "Isaiah Joe", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.367, "underPct": 0.633},
    {"name": "Marvin Bagley III", "line": 7.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.14, "underPct": 0.86},
];const underdogBlocksHitRates = [
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.505, "underPct": 0.495},
];const underdogStealsHitRates = [
    {"name": "Paul George", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.606, "underPct": 0.394},
    {"name": "Shai Gilgeous-Alexander", "line": 1.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.606, "underPct": 0.394},
    {"name": "Cason Wallace", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.579, "underPct": 0.421},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Jaylen Wells", "line": 17.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 39.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 24.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Spencer", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "VJ Edgecombe", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Draymond Green", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 38.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Brunson", "line": 36.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 41.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandin Podziemski", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylin Williams", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gary Payton II", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mikal Bridges", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 39.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 23.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 27.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Buddy Hield", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 22.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Marvin Bagley III", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Quentin Grimes", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Minott", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jose Alvarado", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 34.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 28.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Aaron Wiggins", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dylan Harper", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 21.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Williams", "line": 29.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
];const underdogPRHitRates = [
    {"name": "Keldon Johnson", "line": 19.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 21.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "De'Aaron Fox", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Paul George", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 34.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
];const underdogPAHitRates = [
    {"name": "Anthony Edwards", "line": 33.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Devin Vassell", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paul George", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "De'Aaron Fox", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Williams", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogRAHitRates = [
    {"name": "Saddiq Bey", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Immanuel Quickley", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
];const underdogTurnoversHitRates = [
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const underdogBlocksStealsHitRates = [
    {"name": "Donovan Clingan", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
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

