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
    {"name1": "Ivica Zubac", "name2": "Dillon Brooks", "line1": 15.5, "line2": 18.5, "prediction1": 11.88, "prediction2": 23.81, "side1": "under", "side2": "over", "recommendation": 0, "ev": 69.35, "kelly": 0.347, "sigma1": "Med", "sigma2": "High", "prob1": 0.748, "prob2": 0.77, "hitRate1": 48.0, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 74.6, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Dyson Daniels", "name2": "Austin Reaves", "line1": 12.5, "line2": 22.5, "prediction1": 9.64, "prediction2": 26.89, "side1": "under", "side2": "over", "recommendation": 0, "ev": 56.36, "kelly": 0.282, "sigma1": "Low", "sigma2": "High", "prob1": 0.729, "prob2": 0.73, "hitRate1": 91.3, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 63.5, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "James Harden", "name2": "Rui Hachimura", "line1": 24.5, "line2": 11.5, "prediction1": 20.91, "prediction2": 15.43, "side1": "under", "side2": "over", "recommendation": 0, "ev": 55.76, "kelly": 0.279, "sigma1": "Med", "sigma2": "High", "prob1": 0.727, "prob2": 0.728, "hitRate1": 21.4, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 80.2, "l5_2": 0.6, "l15_2": 0.67},
    {"name1": "Keyonte George", "name2": "Harrison Barnes", "line1": 20.5, "line2": 13.5, "prediction1": 24.8, "prediction2": 16.76, "side1": "over", "side2": "over", "recommendation": 0, "ev": 49.59, "kelly": 0.248, "sigma1": "High", "sigma2": "High", "prob1": 0.728, "prob2": 0.699, "hitRate1": 77.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 56.6, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Deni Avdija", "name2": "Jake LaRavia", "line1": 23.5, "line2": 7.5, "prediction1": 26.88, "prediction2": 11.15, "side1": "over", "side2": "over", "recommendation": 0, "ev": 47.46, "kelly": 0.237, "sigma1": "High", "sigma2": "High", "prob1": 0.697, "prob2": 0.719, "hitRate1": 78.5, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 62.5, "l5_2": 0.2, "l15_2": 0.53},
    {"name1": "Donovan Clingan", "name2": "Kevin Love", "line1": 9.5, "line2": 5.0, "prediction1": 12.47, "prediction2": 7.45, "side1": "over", "side2": "over", "recommendation": 0, "ev": 40.73, "kelly": 0.204, "sigma1": "Med", "sigma2": "Low", "prob1": 0.693, "prob2": 0.691, "hitRate1": 73.5, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 42.7, "l5_2": 0.6, "l15_2": 0.2},
    {"name1": "Brandon Ingram", "name2": "Devin Vassell", "line1": 20.5, "line2": 17.5, "prediction1": 23.75, "prediction2": 14.77, "side1": "over", "side2": "under", "recommendation": 0, "ev": 32.62, "kelly": 0.163, "sigma1": "High", "sigma2": "High", "prob1": 0.682, "prob2": 0.661, "hitRate1": 63.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 93.6, "l5_2": 0.0, "l15_2": 0.07},
    {"name1": "Ryan Kalkbrenner", "name2": "Shai Gilgeous-Alexander", "line1": 8.5, "line2": 31.5, "prediction1": 7.11, "prediction2": 29.12, "side1": "under", "side2": "under", "recommendation": 0, "ev": 24.95, "kelly": 0.125, "sigma1": "Low", "sigma2": "High", "prob1": 0.654, "prob2": 0.65, "hitRate1": 36.6, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 57.3, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Goga Bitadze", "name2": "Marcus Smart", "line1": 6.5, "line2": 6.5, "prediction1": 8.27, "prediction2": 8.76, "side1": "over", "side2": "over", "recommendation": 0, "ev": 23.49, "kelly": 0.117, "sigma1": "Low", "sigma2": "Med", "prob1": 0.649, "prob2": 0.647, "hitRate1": 40.3, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 84.6, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Ziaire Williams", "name2": "Kawhi Leonard", "line1": 10.5, "line2": 18.5, "prediction1": 8.37, "prediction2": 20.38, "side1": "under", "side2": "over", "recommendation": 0, "ev": 19.45, "kelly": 0.097, "sigma1": "High", "sigma2": "Med", "prob1": 0.638, "prob2": 0.637, "hitRate1": 75.0, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 90.3, "l5_2": 0.8, "l15_2": 0.27},
];const prizepicksTriosData = [
    {"name1": "Ivica Zubac", "name2": "Austin Reaves", "name3": "Dillon Brooks", "line1": 15.5, "line2": 22.5, "line3": 18.5, "prediction1": 11.88, "prediction2": 26.89, "prediction3": 23.81, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 126.98, "kelly": 0.254, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.748, "prob2": 0.73, "prob3": 0.77, "hitRate1": 48.0, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 63.5, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 74.6, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "Dyson Daniels", "name2": "James Harden", "name3": "Rui Hachimura", "line1": 12.5, "line2": 24.5, "line3": 11.5, "prediction1": 9.64, "prediction2": 20.91, "prediction3": 15.43, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 108.5, "kelly": 0.217, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "prob1": 0.729, "prob2": 0.727, "prob3": 0.728, "hitRate1": 91.3, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 21.4, "l5_2": 0.8, "l15_2": 0.53, "hitRate3": 80.2, "l5_3": 0.6, "l15_3": 0.67},
    {"name1": "Deni Avdija", "name2": "Keyonte George", "name3": "Harrison Barnes", "line1": 23.5, "line2": 20.5, "line3": 13.5, "prediction1": 26.88, "prediction2": 24.8, "prediction3": 16.76, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 91.61, "kelly": 0.183, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.697, "prob2": 0.728, "prob3": 0.699, "hitRate1": 78.5, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 77.4, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 56.6, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Brandon Ingram", "name2": "Donovan Clingan", "name3": "Jake LaRavia", "line1": 20.5, "line2": 9.5, "line3": 7.5, "prediction1": 23.75, "prediction2": 12.47, "prediction3": 11.15, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 83.55, "kelly": 0.167, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.682, "prob2": 0.693, "prob3": 0.719, "hitRate1": 63.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 73.5, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 62.5, "l5_3": 0.2, "l15_3": 0.53},
    {"name1": "Ryan Kalkbrenner", "name2": "Kevin Love", "name3": "Devin Vassell", "line1": 8.5, "line2": 5.0, "line3": 17.5, "prediction1": 7.11, "prediction2": 7.45, "prediction3": 14.77, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 61.43, "kelly": 0.123, "sigma1": "Low", "sigma2": "Low", "sigma3": "High", "prob1": 0.654, "prob2": 0.691, "prob3": 0.661, "hitRate1": 36.6, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 42.7, "l5_2": 0.6, "l15_2": 0.2, "hitRate3": 93.6, "l5_3": 0.0, "l15_3": 0.07},
    {"name1": "Goga Bitadze", "name2": "Shai Gilgeous-Alexander", "name3": "Marcus Smart", "line1": 6.5, "line2": 31.5, "line3": 6.5, "prediction1": 8.27, "prediction2": 29.12, "prediction3": 8.76, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 47.32, "kelly": 0.095, "sigma1": "Low", "sigma2": "High", "sigma3": "Med", "prob1": 0.649, "prob2": 0.65, "prob3": 0.647, "hitRate1": 40.3, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 57.3, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 84.6, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Anthony Black", "name2": "Ziaire Williams", "name3": "Kawhi Leonard", "line1": 13.5, "line2": 10.5, "line3": 18.5, "prediction1": 15.81, "prediction2": 8.37, "prediction3": 20.38, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 39.73, "kelly": 0.079, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.637, "prob2": 0.638, "prob3": 0.637, "hitRate1": 22.8, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 75.0, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 90.3, "l5_3": 0.8, "l15_3": 0.27},
    {"name1": "Noah Clowney", "name2": "Evan Mobley", "name3": "Ace Bailey", "line1": 12.5, "line2": 19.5, "line3": 11.5, "prediction1": 14.66, "prediction2": 17.57, "prediction3": 9.6, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 33.64, "kelly": 0.067, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.63, "prob2": 0.627, "prob3": 0.626, "hitRate1": 60.7, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 56.9, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 36.2, "l5_3": 0.8, "l15_3": 0.33},
    {"name1": "Onyeka Okongwu", "name2": "Tristan da Silva", "name3": "Jerami Grant", "line1": 18.5, "line2": 13.5, "line3": 18.5, "prediction1": 16.45, "prediction2": 15.52, "prediction3": 16.45, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 29.81, "kelly": 0.06, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.626, "prob2": 0.621, "prob3": 0.619, "hitRate1": 54.1, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 39.6, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 68.1, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Zaccharie Risacher", "name2": "Jamal Shead", "name3": "Darius Garland", "line1": 12.5, "line2": 6.5, "line3": 17.5, "prediction1": 14.34, "prediction2": 5.08, "prediction3": 15.51, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 26.59, "kelly": 0.053, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "prob1": 0.616, "prob2": 0.618, "prob3": 0.616, "hitRate1": 52.2, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 69.1, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 75.2, "l5_3": 0.4, "l15_3": 0.13},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Dyson Daniels", "name2": "Dillon Brooks", "line1": 12.5, "line2": 18.5, "prediction1": 9.64, "prediction2": 23.81, "side1": "under", "side2": "over", "recommendation": 0, "ev": 64.92, "kelly": 0.325, "sigma1": "Low", "sigma2": "High", "prob1": 0.729, "prob2": 0.77, "hitRate1": 91.3, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 74.6, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "James Harden", "name2": "Rui Hachimura", "line1": 24.5, "line2": 11.5, "prediction1": 20.91, "prediction2": 15.43, "side1": "under", "side2": "over", "recommendation": 0, "ev": 55.76, "kelly": 0.279, "sigma1": "Med", "sigma2": "High", "prob1": 0.727, "prob2": 0.728, "hitRate1": 21.4, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 80.2, "l5_2": 0.6, "l15_2": 0.67},
    {"name1": "Keyonte George", "name2": "Harrison Barnes", "line1": 20.5, "line2": 13.5, "prediction1": 24.8, "prediction2": 16.76, "side1": "over", "side2": "over", "recommendation": 0, "ev": 49.59, "kelly": 0.248, "sigma1": "High", "sigma2": "High", "prob1": 0.728, "prob2": 0.699, "hitRate1": 77.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 56.6, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Deni Avdija", "name2": "Kevin Love", "line1": 23.5, "line2": 4.5, "prediction1": 26.88, "prediction2": 7.45, "side1": "over", "side2": "over", "recommendation": 0, "ev": 48.88, "kelly": 0.244, "sigma1": "High", "sigma2": "Low", "prob1": 0.697, "prob2": 0.726, "hitRate1": 78.5, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 60.1, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "Donovan Clingan", "name2": "Jake LaRavia", "line1": 9.5, "line2": 7.5, "prediction1": 12.47, "prediction2": 11.15, "side1": "over", "side2": "over", "recommendation": 0, "ev": 46.44, "kelly": 0.232, "sigma1": "Med", "sigma2": "High", "prob1": 0.693, "prob2": 0.719, "hitRate1": 73.5, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 62.5, "l5_2": 0.2, "l15_2": 0.53},
    {"name1": "Brandon Ingram", "name2": "Shai Gilgeous-Alexander", "line1": 20.5, "line2": 31.5, "prediction1": 23.75, "prediction2": 29.12, "side1": "over", "side2": "under", "recommendation": 0, "ev": 30.32, "kelly": 0.152, "sigma1": "High", "sigma2": "High", "prob1": 0.682, "prob2": 0.65, "hitRate1": 63.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 57.3, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Anthony Black", "name2": "Ziaire Williams", "line1": 13.5, "line2": 10.5, "prediction1": 15.81, "prediction2": 8.37, "side1": "over", "side2": "under", "recommendation": 0, "ev": 19.45, "kelly": 0.097, "sigma1": "High", "sigma2": "High", "prob1": 0.637, "prob2": 0.638, "hitRate1": 22.8, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 75.0, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Noah Clowney", "name2": "Kawhi Leonard", "line1": 12.5, "line2": 18.5, "prediction1": 14.66, "prediction2": 20.38, "side1": "over", "side2": "over", "recommendation": 0, "ev": 17.89, "kelly": 0.089, "sigma1": "High", "sigma2": "Med", "prob1": 0.63, "prob2": 0.637, "hitRate1": 60.7, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 90.3, "l5_2": 0.8, "l15_2": 0.27},
    {"name1": "Evan Mobley", "name2": "Ace Bailey", "line1": 19.5, "line2": 11.5, "prediction1": 17.57, "prediction2": 9.6, "side1": "under", "side2": "under", "recommendation": 0, "ev": 15.55, "kelly": 0.078, "sigma1": "Med", "sigma2": "Med", "prob1": 0.627, "prob2": 0.626, "hitRate1": 56.9, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 36.2, "l5_2": 0.8, "l15_2": 0.33},
    {"name1": "Jalen Johnson", "name2": "Jerami Grant", "line1": 22.5, "line2": 18.5, "prediction1": 24.5, "prediction2": 16.45, "side1": "over", "side2": "under", "recommendation": 0, "ev": 13.0, "kelly": 0.065, "sigma1": "High", "sigma2": "High", "prob1": 0.621, "prob2": 0.619, "hitRate1": 57.6, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 68.1, "l5_2": 0.4, "l15_2": 0.4},
];const underdogTriosData = [
    {"name1": "Dyson Daniels", "name2": "Rui Hachimura", "name3": "Dillon Brooks", "line1": 12.5, "line2": 11.5, "line3": 18.5, "prediction1": 9.64, "prediction2": 15.43, "prediction3": 23.81, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 120.65, "kelly": 0.241, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.729, "prob2": 0.728, "prob3": 0.77, "hitRate1": 91.3, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 80.2, "l5_2": 0.6, "l15_2": 0.67, "hitRate3": 74.6, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "James Harden", "name2": "Keyonte George", "name3": "Kevin Love", "line1": 24.5, "line2": 20.5, "line3": 4.5, "prediction1": 20.91, "prediction2": 24.8, "prediction3": 7.45, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 107.58, "kelly": 0.215, "sigma1": "Med", "sigma2": "High", "sigma3": "Low", "prob1": 0.727, "prob2": 0.728, "prob3": 0.726, "hitRate1": 21.4, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 77.4, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 60.1, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Deni Avdija", "name2": "Jake LaRavia", "name3": "Harrison Barnes", "line1": 23.5, "line2": 7.5, "line3": 13.5, "prediction1": 26.88, "prediction2": 11.15, "prediction3": 16.76, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 89.33, "kelly": 0.179, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.697, "prob2": 0.719, "prob3": 0.699, "hitRate1": 78.5, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 62.5, "l5_2": 0.2, "l15_2": 0.53, "hitRate3": 56.6, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Anthony Black", "name2": "Brandon Ingram", "name3": "Donovan Clingan", "line1": 13.5, "line2": 20.5, "line3": 9.5, "prediction1": 15.81, "prediction2": 23.75, "prediction3": 12.47, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 62.53, "kelly": 0.125, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.637, "prob2": 0.682, "prob3": 0.693, "hitRate1": 22.8, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 63.7, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 73.5, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Ziaire Williams", "name2": "Kawhi Leonard", "name3": "Shai Gilgeous-Alexander", "line1": 10.5, "line2": 18.5, "line3": 31.5, "prediction1": 8.37, "prediction2": 20.38, "prediction3": 29.12, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 42.5, "kelly": 0.085, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.638, "prob2": 0.637, "prob3": 0.65, "hitRate1": 75.0, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 90.3, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 57.3, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Noah Clowney", "name2": "Evan Mobley", "name3": "Ace Bailey", "line1": 12.5, "line2": 19.5, "line3": 11.5, "prediction1": 14.66, "prediction2": 17.57, "prediction3": 9.6, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 33.64, "kelly": 0.067, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.63, "prob2": 0.627, "prob3": 0.626, "hitRate1": 60.7, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 56.9, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 36.2, "l5_3": 0.8, "l15_3": 0.33},
    {"name1": "Jalen Johnson", "name2": "Jamal Shead", "name3": "Jerami Grant", "line1": 22.5, "line2": 6.5, "line3": 18.5, "prediction1": 24.5, "prediction2": 5.08, "prediction3": 16.45, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 28.23, "kelly": 0.056, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "prob1": 0.621, "prob2": 0.618, "prob3": 0.619, "hitRate1": 57.6, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 69.1, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 68.1, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Donovan Mitchell", "name2": "Darius Garland", "name3": "Kyle Filipowski", "line1": 27.5, "line2": 17.5, "line3": 8.5, "prediction1": 29.0, "prediction2": 15.51, "prediction3": 9.62, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 14.32, "kelly": 0.029, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.588, "prob2": 0.616, "prob3": 0.584, "hitRate1": 63.5, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 75.2, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 59.4, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Onyeka Okongwu", "name2": "Luka Don\u010di\u0107", "name3": "Collin Gillespie", "line1": 17.5, "line2": 31.5, "line3": 13.5, "prediction1": 16.45, "prediction2": 32.42, "prediction3": 12.16, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": -0.18, "kelly": 0.0, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.565, "prob2": 0.565, "prob3": 0.579, "hitRate1": 44.7, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 65.1, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Franz Wagner", "name2": "Lauri Markkanen", "name3": "Svi Mykhailiuk", "line1": 25.5, "line2": 26.5, "line3": 8.5, "prediction1": 24.7, "prediction2": 27.39, "prediction3": 9.24, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": -10.53, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.548, "prob2": 0.552, "prob3": 0.548, "hitRate1": 68.3, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 75.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 69.4, "l5_3": 0.6, "l15_3": 0.6},
];const prizepicksPointsHitRates = [
    {"name": "Kawhi Leonard", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.903, "underPct": 0.097},
    {"name": "Marcus Smart", "line": 6.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.846, "underPct": 0.154},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.802, "underPct": 0.198},
    {"name": "James Harden", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.786, "underPct": 0.214},
    {"name": "Deni Avdija", "line": 23.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.785, "underPct": 0.215},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.774, "underPct": 0.226},
    {"name": "Nickeil Alexander-Walker", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.767, "underPct": 0.233},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.746, "underPct": 0.254},
    {"name": "Jaylon Tyson", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.739, "underPct": 0.261},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.735, "underPct": 0.265},
    {"name": "Immanuel Quickley", "line": 16.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.725, "underPct": 0.275},
    {"name": "Deandre Ayton", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.708, "underPct": 0.292},
    {"name": "Svi Mykhailiuk", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.694, "underPct": 0.306},
    {"name": "Luka Garza", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.683, "underPct": 0.317},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.678, "underPct": 0.322},
    {"name": "Jordan Goodwin", "line": 8.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.64, "underPct": 0.36},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.638, "underPct": 0.362},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.638, "underPct": 0.362},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.637, "underPct": 0.363},
    {"name": "Donovan Mitchell", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.635, "underPct": 0.365},
    {"name": "Austin Reaves", "line": 22.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.635, "underPct": 0.365},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.634, "underPct": 0.366},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.63, "underPct": 0.37},
    {"name": "Isaiah Collier", "line": 8.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.63, "underPct": 0.37},
    {"name": "LeBron James", "line": 18.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.627, "underPct": 0.373},
    {"name": "Jake LaRavia", "line": 7.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.625, "underPct": 0.375},
    {"name": "Noah Clowney", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.607, "underPct": 0.393},
    {"name": "De'Andre Hunter", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.601, "underPct": 0.399},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.594, "underPct": 0.406},
    {"name": "Day'Ron Sharpe", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.59, "underPct": 0.41},
    {"name": "Derrick White", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.586, "underPct": 0.414},
    {"name": "Jaylen Brown", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.566, "underPct": 0.434},
    {"name": "Harrison Barnes", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.566, "underPct": 0.434},
    {"name": "Miles Bridges", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.546, "underPct": 0.454},
    {"name": "Brice Sensabaugh", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.528, "underPct": 0.472},
    {"name": "Zaccharie Risacher", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.522, "underPct": 0.478},
    {"name": "Ivica Zubac", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.52, "underPct": 0.48},
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.509, "underPct": 0.491},
    {"name": "De'Aaron Fox", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.502, "underPct": 0.498},
    {"name": "Jalen Johnson", "line": 23.5, "l5": 1.0, "l10": 0.5, "l15": 0.4, "overPct": 0.494, "underPct": 0.506},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.485, "underPct": 0.515},
    {"name": "Anfernee Simons", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.478, "underPct": 0.522},
    {"name": "Isaiah Joe", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.476, "underPct": 0.524},
    {"name": "Mark Williams", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.46, "underPct": 0.54},
    {"name": "Onyeka Okongwu", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.459, "underPct": 0.541},
    {"name": "Kon Knueppel", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.436, "underPct": 0.564},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.427, "underPct": 0.573},
    {"name": "Kevin Love", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.427, "underPct": 0.573},
    {"name": "Ajay Mitchell", "line": 15.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.426, "underPct": 0.574},
    {"name": "Luke Kornet", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.417, "underPct": 0.583},
    {"name": "Drake Powell", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.403, "underPct": 0.597},
    {"name": "Goga Bitadze", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.403, "underPct": 0.597},
    {"name": "Royce O'Neale", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.397, "underPct": 0.603},
    {"name": "Tristan da Silva", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.396, "underPct": 0.604},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.379, "underPct": 0.621},
    {"name": "Kris Dunn", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.35, "underPct": 0.65},
    {"name": "Collin Gillespie", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.349, "underPct": 0.651},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.319, "underPct": 0.681},
    {"name": "Franz Wagner", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.309, "underPct": 0.691},
    {"name": "Ryan Dunn", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.306, "underPct": 0.694},
    {"name": "Luka Doncic", "line": 31.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.298, "underPct": 0.702},
    {"name": "Keldon Johnson", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.293, "underPct": 0.707},
    {"name": "Desmond Bane", "line": 21.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.275, "underPct": 0.725},
    {"name": "Kris Murray", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.254, "underPct": 0.746},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.253, "underPct": 0.747},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.25, "underPct": 0.75},
    {"name": "Darius Garland", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.248, "underPct": 0.752},
    {"name": "Tyrese Martin", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.231, "underPct": 0.769},
    {"name": "Anthony Black", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.228, "underPct": 0.772},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.225, "underPct": 0.775},
    {"name": "Dean Wade", "line": 6.0, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.222, "underPct": 0.778},
    {"name": "Brook Lopez", "line": 6.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.213, "underPct": 0.787},
    {"name": "Devin Booker", "line": 29.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.18, "underPct": 0.82},
    {"name": "John Collins", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.159, "underPct": 0.841},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.087, "underPct": 0.913},
    {"name": "Sion James", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.08, "underPct": 0.92},
    {"name": "Collin Sexton", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.27, "overPct": 0.079, "underPct": 0.921},
    {"name": "Devin Vassell", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.064, "underPct": 0.936},
];const prizepicksAssistsHitRates = [
    {"name": "LeBron James", "line": 7.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.722, "underPct": 0.278},
    {"name": "Kawhi Leonard", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.692, "underPct": 0.308},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.651, "underPct": 0.349},
    {"name": "Derrick White", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.564, "underPct": 0.436},
    {"name": "Jalen Johnson", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.54, "underPct": 0.46},
    {"name": "Austin Reaves", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.534, "underPct": 0.466},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.513, "underPct": 0.487},
    {"name": "Keyonte George", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.502, "underPct": 0.498},
    {"name": "Jordan Goodwin", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.502, "underPct": 0.498},
    {"name": "Zaccharie Risacher", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.499, "underPct": 0.501},
    {"name": "Donovan Mitchell", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.494, "underPct": 0.506},
    {"name": "Jamal Shead", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.49, "underPct": 0.51},
    {"name": "Jake LaRavia", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.461, "underPct": 0.539},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.444, "underPct": 0.556},
    {"name": "Isaiah Collier", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.443, "underPct": 0.557},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.427, "underPct": 0.573},
    {"name": "Franz Wagner", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.426, "underPct": 0.574},
    {"name": "Deni Avdija", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.412, "underPct": 0.588},
    {"name": "Shai Gilgeous-Alexander", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.384, "underPct": 0.616},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.373, "underPct": 0.627},
    {"name": "Julian Champagnie", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.365, "underPct": 0.635},
    {"name": "De'Aaron Fox", "line": 7.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.355, "underPct": 0.645},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.345, "underPct": 0.655},
    {"name": "Darius Garland", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.344, "underPct": 0.656},
    {"name": "Jaylen Brown", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.312, "underPct": 0.688},
    {"name": "James Harden", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.295, "underPct": 0.705},
    {"name": "Collin Gillespie", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.281, "underPct": 0.719},
    {"name": "Luka Doncic", "line": 9.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.057, "underPct": 0.943},
];const prizepicksReboundsHitRates = [
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.704, "underPct": 0.296},
    {"name": "Donovan Mitchell", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.689, "underPct": 0.311},
    {"name": "James Harden", "line": 5.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.687, "underPct": 0.313},
    {"name": "Kawhi Leonard", "line": 4.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.681, "underPct": 0.319},
    {"name": "Franz Wagner", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.622, "underPct": 0.378},
    {"name": "Deandre Ayton", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.615, "underPct": 0.385},
    {"name": "Jalen Johnson", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.559, "underPct": 0.441},
    {"name": "Kon Knueppel", "line": 5.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.526, "underPct": 0.474},
    {"name": "Luka Garza", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.524, "underPct": 0.476},
    {"name": "Jerami Grant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.521, "underPct": 0.479},
    {"name": "Austin Reaves", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "Collin Gillespie", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.496, "underPct": 0.504},
    {"name": "Shai Gilgeous-Alexander", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.49, "underPct": 0.51},
    {"name": "Kyle Filipowski", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.487, "underPct": 0.513},
    {"name": "Scottie Barnes", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.483, "underPct": 0.517},
    {"name": "Luguentz Dort", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Lauri Markkanen", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.476, "underPct": 0.524},
    {"name": "Isaiah Hartenstein", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.476, "underPct": 0.524},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.455, "underPct": 0.545},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.448, "underPct": 0.552},
    {"name": "Tristan da Silva", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.444, "underPct": 0.556},
    {"name": "Keldon Johnson", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.439, "underPct": 0.561},
    {"name": "Jaylon Tyson", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.431, "underPct": 0.569},
    {"name": "Royce O'Neale", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.431, "underPct": 0.569},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.428, "underPct": 0.572},
    {"name": "Kevin Love", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.428, "underPct": 0.572},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.425, "underPct": 0.575},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.423, "underPct": 0.577},
    {"name": "Immanuel Quickley", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.422, "underPct": 0.578},
    {"name": "Dean Wade", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.408, "underPct": 0.592},
    {"name": "Deni Avdija", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.391, "underPct": 0.609},
    {"name": "Devin Booker", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.386, "underPct": 0.614},
    {"name": "LeBron James", "line": 6.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.385, "underPct": 0.615},
    {"name": "Kris Murray", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.384, "underPct": 0.616},
    {"name": "Chet Holmgren", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.364, "underPct": 0.636},
    {"name": "Terance Mann", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.363, "underPct": 0.637},
    {"name": "Derrick White", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.344, "underPct": 0.656},
    {"name": "Miles Bridges", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.327, "underPct": 0.673},
    {"name": "Gradey Dick", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.304, "underPct": 0.696},
    {"name": "Brandon Ingram", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.298, "underPct": 0.702},
    {"name": "De'Andre Hunter", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.298, "underPct": 0.702},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.273, "underPct": 0.727},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.255, "underPct": 0.745},
    {"name": "Harrison Barnes", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.247, "underPct": 0.753},
    {"name": "Goga Bitadze", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.247, "underPct": 0.753},
    {"name": "John Collins", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.246, "underPct": 0.754},
    {"name": "Luke Kornet", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.243, "underPct": 0.757},
    {"name": "Mark Williams", "line": 8.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.208, "underPct": 0.792},
    {"name": "Devin Vassell", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.191, "underPct": 0.809},
    {"name": "Ziaire Williams", "line": 3.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.183, "underPct": 0.817},
    {"name": "Jeremy Sochan", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.163, "underPct": 0.837},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.13, "underPct": 0.87},
];const prizepicksBlocksHitRates = [
    {"name": "Ryan Kalkbrenner", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.64, "underPct": 0.36},
    {"name": "Anthony Black", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.383, "underPct": 0.617},
    {"name": "Jaylin Williams", "line": 0.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.392, "underPct": 0.608},
    {"name": "Luka Doncic", "line": 0.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.143, "underPct": 0.857},
    {"name": "Ryan Dunn", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.406, "underPct": 0.594},
];const prizepicksStealsHitRates = [
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.472, "underPct": 0.528},
    {"name": "Jalen Johnson", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.56, "underPct": 0.44},
    {"name": "Drake Powell", "line": 0.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.628, "underPct": 0.372},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.524, "underPct": 0.476},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.581, "underPct": 0.419},
    {"name": "Sidy Cissoko", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.482, "underPct": 0.518},
    {"name": "Luka Doncic", "line": 1.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.265, "underPct": 0.735},
    {"name": "Brice Sensabaugh", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.454, "underPct": 0.546},
    {"name": "Jeremy Sochan", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.362, "underPct": 0.638},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Jalen Johnson", "line": 40.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tristan da Silva", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Love", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 16.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Filipowski", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 34.0, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alex Caruso", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylon Tyson", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 37.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 38.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylin Williams", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deandre Ayton", "line": 24.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 31.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brice Sensabaugh", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 34.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Joe", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 43.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luka Garza", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Day'Ron Sharpe", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Onyeka Okongwu", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Marcus Smart", "line": 10.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Dunn", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 14.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremy Sochan", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dean Wade", "line": 12.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 36.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sidy Cissoko", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zaccharie Risacher", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julian Champagnie", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 41.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 32.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Vassell", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jake LaRavia", "line": 12.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Doncic", "line": 48.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Collin Sexton", "line": 24.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "LeBron James", "line": 32.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Jordan Goodwin", "line": 12.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 23.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Filipowski", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 27.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Donovan Mitchell", "line": 32.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "James Harden", "line": 29.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Isaiah Collier", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rui Hachimura", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylon Tyson", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylin Williams", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sidy Cissoko", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brice Sensabaugh", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 15.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Garza", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zaccharie Risacher", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Neemias Queta", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jeremy Sochan", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Onyeka Okongwu", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Joe", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Darius Garland", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 19.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jake LaRavia", "line": 11.0, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sion James", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luka Doncic", "line": 39.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Dunn", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mark Williams", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 16.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 25.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.0, "l10": 0.0, "l15": 0.13, "overPct": 0.0, "underPct": 1.0},
    {"name": "Collin Sexton", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brook Lopez", "line": 8.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPAHitRates = [
    {"name": "Jalen Johnson", "line": 31.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kawhi Leonard", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luka Garza", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "James Harden", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Goodwin", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brice Sensabaugh", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Harrison Barnes", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Chet Holmgren", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 15.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Gillespie", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 28.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rui Hachimura", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Dunn", "line": 8.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tristan da Silva", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Noah Clowney", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremy Sochan", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zaccharie Risacher", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Filipowski", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Dunn", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Marcus Smart", "line": 9.0, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Scottie Barnes", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sidy Cissoko", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brook Lopez", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "John Collins", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Doncic", "line": 40.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 37.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jake LaRavia", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sion James", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dean Wade", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 22.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "LeBron James", "line": 26.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksRAHitRates = [
    {"name": "Dillon Brooks", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 10.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 7.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 10.0, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 7.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Filipowski", "line": 8.0, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brice Sensabaugh", "line": 3.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Caruso", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 11.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deandre Ayton", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 17.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 13.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zaccharie Risacher", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cason Wallace", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremy Sochan", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keldon Johnson", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Love", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Garza", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Darius Garland", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Hartenstein", "line": 13.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sidy Cissoko", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "James Harden", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 14.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brook Lopez", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 10.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julian Champagnie", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 8.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 12.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Day'Ron Sharpe", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Immanuel Quickley", "line": 11.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 7.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles Bridges", "line": 11.0, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 16.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksTurnoversHitRates = [
    {"name": "Anfernee Simons", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Day'Ron Sharpe", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Caruso", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Filipowski", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rui Hachimura", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ivica Zubac", "line": 1.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brook Lopez", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luka Doncic", "line": 3.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Devin Vassell", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jake LaRavia", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogPointsHitRates = [
    {"name": "Kawhi Leonard", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.903, "underPct": 0.097},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.802, "underPct": 0.198},
    {"name": "James Harden", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.786, "underPct": 0.214},
    {"name": "Deni Avdija", "line": 23.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.785, "underPct": 0.215},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.774, "underPct": 0.226},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.757, "underPct": 0.243},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.746, "underPct": 0.254},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.735, "underPct": 0.265},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.694, "underPct": 0.306},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.678, "underPct": 0.322},
    {"name": "Jordan Goodwin", "line": 8.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.64, "underPct": 0.36},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.638, "underPct": 0.362},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.638, "underPct": 0.362},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.637, "underPct": 0.363},
    {"name": "Donovan Mitchell", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.635, "underPct": 0.365},
    {"name": "LeBron James", "line": 18.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.627, "underPct": 0.373},
    {"name": "Jake LaRavia", "line": 7.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.625, "underPct": 0.375},
    {"name": "Noah Clowney", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.607, "underPct": 0.393},
    {"name": "Kevin Love", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.601, "underPct": 0.399},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.594, "underPct": 0.406},
    {"name": "Derrick White", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.586, "underPct": 0.414},
    {"name": "Jalen Johnson", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.576, "underPct": 0.424},
    {"name": "Harrison Barnes", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.566, "underPct": 0.434},
    {"name": "Onyeka Okongwu", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.553, "underPct": 0.447},
    {"name": "Brice Sensabaugh", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.53, "underPct": 0.47},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.485, "underPct": 0.515},
    {"name": "Isaiah Joe", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.476, "underPct": 0.524},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.427, "underPct": 0.573},
    {"name": "Ajay Mitchell", "line": 15.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.426, "underPct": 0.574},
    {"name": "Royce O'Neale", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.397, "underPct": 0.603},
    {"name": "Collin Gillespie", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.349, "underPct": 0.651},
    {"name": "De'Aaron Fox", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.344, "underPct": 0.656},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.319, "underPct": 0.681},
    {"name": "Franz Wagner", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.309, "underPct": 0.691},
    {"name": "Luka Doncic", "line": 31.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.298, "underPct": 0.702},
    {"name": "Desmond Bane", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.275, "underPct": 0.725},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.253, "underPct": 0.747},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.25, "underPct": 0.75},
    {"name": "Darius Garland", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.248, "underPct": 0.752},
    {"name": "Anthony Black", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.228, "underPct": 0.772},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.225, "underPct": 0.775},
    {"name": "Brook Lopez", "line": 6.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.213, "underPct": 0.787},
    {"name": "Devin Booker", "line": 29.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.18, "underPct": 0.82},
    {"name": "Terance Mann", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.146, "underPct": 0.854},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.087, "underPct": 0.913},
];const underdogAssistsHitRates = [
    {"name": "Jamal Shead", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.66, "underPct": 0.34},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.513, "underPct": 0.487},
    {"name": "Jordan Goodwin", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.502, "underPct": 0.498},
    {"name": "Zaccharie Risacher", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.499, "underPct": 0.501},
    {"name": "Jake LaRavia", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.461, "underPct": 0.539},
    {"name": "Julian Champagnie", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.365, "underPct": 0.635},
    {"name": "Jeremy Sochan", "line": 2.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.351, "underPct": 0.649},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.345, "underPct": 0.655},
];const underdogReboundsHitRates = [
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.704, "underPct": 0.296},
    {"name": "Jalen Johnson", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.559, "underPct": 0.441},
    {"name": "Luka Garza", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.524, "underPct": 0.476},
    {"name": "Collin Gillespie", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.496, "underPct": 0.504},
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.488, "underPct": 0.512},
    {"name": "Luguentz Dort", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Isaiah Hartenstein", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.476, "underPct": 0.524},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.455, "underPct": 0.545},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.425, "underPct": 0.575},
    {"name": "Alex Caruso", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.377, "underPct": 0.623},
    {"name": "Chet Holmgren", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.364, "underPct": 0.636},
    {"name": "Gradey Dick", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.304, "underPct": 0.696},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.255, "underPct": 0.745},
    {"name": "Luke Kornet", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.243, "underPct": 0.757},
    {"name": "Julian Champagnie", "line": 4.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.134, "underPct": 0.866},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.472, "underPct": 0.528},
    {"name": "Anthony Black", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.447, "underPct": 0.553},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.581, "underPct": 0.419},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Jalen Johnson", "line": 40.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyle Filipowski", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Love", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 34.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Caruso", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 37.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 38.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 43.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylin Williams", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Dunn", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brice Sensabaugh", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keldon Johnson", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 36.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Royce O'Neale", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Marcus Smart", "line": 10.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Dunn", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Evan Mobley", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Day'Ron Sharpe", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luguentz Dort", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sidy Cissoko", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ajay Mitchell", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 41.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 49.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jake LaRavia", "line": 12.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 32.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPRHitRates = [
    {"name": "Jalen Johnson", "line": 32.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 23.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Donovan Clingan", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 32.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 29.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Deni Avdija", "line": 30.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Hartenstein", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 39.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 25.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.0, "l10": 0.0, "l15": 0.13, "overPct": 0.0, "underPct": 1.0},
];const underdogPAHitRates = [
    {"name": "Jalen Johnson", "line": 30.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "James Harden", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 32.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Austin Reaves", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Chet Holmgren", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luka Doncic", "line": 40.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 37.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "LeBron James", "line": 26.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogRAHitRates = [
    {"name": "Keyonte George", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremy Sochan", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Nickeil Alexander-Walker", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "James Harden", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 9.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
];const underdogTurnoversHitRates = [
    {"name": "Desmond Bane", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Darius Garland", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Sexton", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Doncic", "line": 3.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogBlocksStealsHitRates = [
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
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
    
    if (currentType === 'singles') {
        statsHTML = `
            <div class="stat-card">
                <div class="stat-label">Projection</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Model's projected value given the context of the game and player performance</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Expected Value %</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value as a percentage of your stake (Ex. If EV% is 5%, you can expect to profit 5% of whatever stake you place on that bet on average.)</div>
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

