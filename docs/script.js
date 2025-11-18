const prizepicksSinglesData = [
    {"name": "Josh Giddey", "bookmaker": "Bovada", "line": 21.5, "prediction": 25.38, "side": "Over", "odds": 200, "recommendation": 0, "ev": 10.56, "roi": 105.6, "kelly": 0.528, "sigma": "High"},
    {"name": "Simone Fontecchio", "bookmaker": "Bovada", "line": 11.5, "prediction": 13.35, "side": "Over", "odds": 205, "recommendation": 0, "ev": 9.28, "roi": 92.8, "kelly": 0.453, "sigma": "High"},
    {"name": "Josh Giddey", "bookmaker": "Bovada", "line": 20.5, "prediction": 25.38, "side": "Over", "odds": 160, "recommendation": 1, "ev": 8.93, "roi": 89.3, "kelly": 0.558, "sigma": "High"},
    {"name": "Naji Marshall", "bookmaker": "Bovada", "line": 12.5, "prediction": 15.16, "side": "Over", "odds": 180, "recommendation": 0, "ev": 8.42, "roi": 84.2, "kelly": 0.468, "sigma": "High"},
    {"name": "LaMelo Ball", "bookmaker": "Bovada", "line": 23.5, "prediction": 25.13, "side": "Over", "odds": 205, "recommendation": 0, "ev": 7.83, "roi": 78.3, "kelly": 0.382, "sigma": "High"},
    {"name": "Nikola Jokic", "bookmaker": "Bovada", "line": 23.5, "prediction": 22.11, "side": "Under", "odds": 200, "recommendation": 0, "ev": 7.67, "roi": 76.7, "kelly": 0.384, "sigma": "High"},
    {"name": "Josh Giddey", "bookmaker": "Bovada", "line": 19.5, "prediction": 25.38, "side": "Over", "odds": 130, "recommendation": 1, "ev": 7.54, "roi": 75.4, "kelly": 0.58, "sigma": "High"},
    {"name": "Simone Fontecchio", "bookmaker": "Bovada", "line": 10.5, "prediction": 13.35, "side": "Over", "odds": 155, "recommendation": 0, "ev": 7.3, "roi": 73.0, "kelly": 0.471, "sigma": "High"},
    {"name": "Brandon Williams", "bookmaker": "Bovada", "line": 16.5, "prediction": 18.22, "side": "Over", "odds": 190, "recommendation": 0, "ev": 7.22, "roi": 72.2, "kelly": 0.38, "sigma": "High"},
    {"name": "Nikola Jokic", "bookmaker": "Bovada", "line": 24.5, "prediction": 22.11, "side": "Under", "odds": 165, "recommendation": 0, "ev": 7.2, "roi": 72.0, "kelly": 0.436, "sigma": "High"},
    {"name": "Cooper Flagg", "bookmaker": "Bovada", "line": 18.5, "prediction": 20.98, "side": "Over", "odds": 165, "recommendation": 0, "ev": 7.08, "roi": 70.8, "kelly": 0.429, "sigma": "High"},
    {"name": "LaMelo Ball", "bookmaker": "Bovada", "line": 22.5, "prediction": 25.13, "side": "Over", "odds": 165, "recommendation": 0, "ev": 6.88, "roi": 68.8, "kelly": 0.417, "sigma": "High"},
    {"name": "Zion Williamson", "bookmaker": "Bovada", "line": 21.5, "prediction": 22.72, "side": "Over", "odds": 185, "recommendation": 0, "ev": 6.67, "roi": 66.7, "kelly": 0.36, "sigma": "High"},
    {"name": "Nikola Jokic", "bookmaker": "Bovada", "line": 25.5, "prediction": 22.11, "side": "Under", "odds": 135, "recommendation": 0, "ev": 6.55, "roi": 65.5, "kelly": 0.485, "sigma": "High"},
    {"name": "Josh Giddey", "bookmaker": "BetRivers", "line": 18.5, "prediction": 25.38, "side": "Over", "odds": 105, "recommendation": 1, "ev": 6.41, "roi": 64.1, "kelly": 0.61, "sigma": "High"},
];const prizepicksPairsData = [
    {"name1": "Nikola Joki\u0107", "name2": "Josh Giddey", "line1": 0.5, "line2": 17.5, "prediction1": 22.11, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 1, "ev": 12.55, "kelly": 0.628, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 85.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Simone Fontecchio", "name2": "Nikola Joki\u0107", "line1": 8.5, "line2": 0.5, "prediction1": 13.35, "prediction2": 22.11, "side1": "over", "side2": "over", "recommendation": 1, "ev": 11.61, "kelly": 0.581, "sigma1": "High", "sigma2": "High", "hitRate1": 69.9, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Zion Williamson", "name2": "Nikola Joki\u0107", "line1": 17.5, "line2": 0.5, "prediction1": 22.72, "prediction2": 22.11, "side1": "over", "side2": "over", "recommendation": 1, "ev": 11.31, "kelly": 0.566, "sigma1": "High", "sigma2": "High", "hitRate1": 86.1, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Simone Fontecchio", "name2": "Josh Giddey", "line1": 8.5, "line2": 17.5, "prediction1": 13.35, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 1, "ev": 8.05, "kelly": 0.402, "sigma1": "High", "sigma2": "High", "hitRate1": 69.9, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 85.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Zion Williamson", "name2": "Josh Giddey", "line1": 17.5, "line2": 17.5, "prediction1": 22.72, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 1, "ev": 8.02, "kelly": 0.401, "sigma1": "High", "sigma2": "High", "hitRate1": 86.1, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 85.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Simone Fontecchio", "name2": "Zion Williamson", "line1": 8.5, "line2": 17.5, "prediction1": 13.35, "prediction2": 22.72, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.07, "kelly": 0.354, "sigma1": "High", "sigma2": "High", "hitRate1": 69.9, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 86.1, "l5_2": 0.8, "l15_2": 0.27},
    {"name1": "LaMelo Ball", "name2": "Naji Marshall", "line1": 19.5, "line2": 10.0, "prediction1": 25.13, "prediction2": 15.16, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.14, "kelly": 0.307, "sigma1": "High", "sigma2": "High", "hitRate1": 69.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 86.0, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "LaMelo Ball", "name2": "Cooper Flagg", "line1": 19.5, "line2": 15.5, "prediction1": 25.13, "prediction2": 20.98, "side1": "over", "side2": "over", "recommendation": 1, "ev": 5.96, "kelly": 0.298, "sigma1": "High", "sigma2": "High", "hitRate1": 69.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 61.2, "l5_2": 0.8, "l15_2": 0.53},
    {"name1": "Jarace Walker", "name2": "Cooper Flagg", "line1": 9.5, "line2": 15.5, "prediction1": 13.75, "prediction2": 20.98, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.45, "kelly": 0.273, "sigma1": "High", "sigma2": "High", "hitRate1": 42.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 61.2, "l5_2": 0.8, "l15_2": 0.53},
    {"name1": "Jarace Walker", "name2": "LaMelo Ball", "line1": 9.5, "line2": 19.5, "prediction1": 13.75, "prediction2": 25.13, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.38, "kelly": 0.269, "sigma1": "High", "sigma2": "High", "hitRate1": 42.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 69.4, "l5_2": 0.4, "l15_2": 0.27},
];const prizepicksTriosData = [
    {"name1": "Simone Fontecchio", "name2": "Nikola Joki\u0107", "name3": "Josh Giddey", "line1": 8.5, "line2": 0.5, "line3": 17.5, "prediction1": 13.35, "prediction2": 22.11, "prediction3": 25.38, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 22.56, "kelly": 0.451, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 69.9, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 85.0, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Zion Williamson", "name2": "Nikola Joki\u0107", "name3": "Josh Giddey", "line1": 17.5, "line2": 0.5, "line3": 17.5, "prediction1": 22.72, "prediction2": 22.11, "prediction3": 25.38, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 21.98, "kelly": 0.44, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 86.1, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 85.0, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "LaMelo Ball", "name2": "Simone Fontecchio", "name3": "Zion Williamson", "line1": 19.5, "line2": 8.5, "line3": 17.5, "prediction1": 25.13, "prediction2": 13.35, "prediction3": 22.72, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 14.02, "kelly": 0.28, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 69.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 69.9, "l5_2": 0.8, "l15_2": 0.73, "hitRate3": 86.1, "l5_3": 0.8, "l15_3": 0.27},
    {"name1": "LaMelo Ball", "name2": "Cooper Flagg", "name3": "Naji Marshall", "line1": 19.5, "line2": 15.5, "line3": 10.0, "prediction1": 25.13, "prediction2": 20.98, "prediction3": 15.16, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 11.96, "kelly": 0.239, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 69.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 61.2, "l5_2": 0.8, "l15_2": 0.53, "hitRate3": 86.0, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "Jarace Walker", "name2": "Cooper Flagg", "name3": "Naji Marshall", "line1": 9.5, "line2": 15.5, "line3": 10.0, "prediction1": 13.75, "prediction2": 20.98, "prediction3": 15.16, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.15, "kelly": 0.223, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 42.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 61.2, "l5_2": 0.8, "l15_2": 0.53, "hitRate3": 86.0, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "Jarace Walker", "name2": "Tony Bradley", "name3": "Brandon Williams", "line1": 9.5, "line2": 4.5, "line3": 13.5, "prediction1": 13.75, "prediction2": 6.45, "prediction3": 18.22, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.18, "kelly": 0.184, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "hitRate1": 42.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 71.8, "l5_2": 1.0, "l15_2": 0.6, "hitRate3": 36.6, "l5_3": 0.8, "l15_3": 0.33},
    {"name1": "Tony Bradley", "name2": "Brandon Williams", "name3": "D'Angelo Russell", "line1": 4.5, "line2": 13.5, "line3": 11.5, "prediction1": 6.45, "prediction2": 18.22, "prediction3": 15.64, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.18, "kelly": 0.164, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 71.8, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 36.6, "l5_2": 0.8, "l15_2": 0.33, "hitRate3": 36.1, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Bennedict Mathurin", "name2": "Brook Lopez", "name3": "D'Angelo Russell", "line1": 16.5, "line2": 6.5, "line3": 11.5, "prediction1": 19.12, "prediction2": 9.01, "prediction3": 15.64, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.07, "kelly": 0.141, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "hitRate1": 83.5, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 56.1, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 36.1, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Brook Lopez", "name2": "Collin Murray-Boyles", "name3": "Alex Caruso", "line1": 6.5, "line2": 6.5, "line3": 5.5, "prediction1": 9.01, "prediction2": 8.84, "prediction3": 7.32, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.8, "kelly": 0.136, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "hitRate1": 56.1, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 71.9, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 55.9, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Collin Murray-Boyles", "name2": "Gradey Dick", "name3": "Alex Caruso", "line1": 6.5, "line2": 6.5, "line3": 5.5, "prediction1": 8.84, "prediction2": 8.56, "prediction3": 7.32, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.37, "kelly": 0.127, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "hitRate1": 71.9, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 70.4, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 55.9, "l5_3": 0.4, "l15_3": 0.4},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Nikola Joki\u0107", "name2": "Josh Giddey", "line1": 27.5, "line2": 18.5, "prediction1": 22.11, "prediction2": 25.38, "side1": "under", "side2": "over", "recommendation": 1, "ev": 7.53, "kelly": 0.376, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Zion Williamson", "name2": "Josh Giddey", "line1": 17.5, "line2": 18.5, "prediction1": 22.72, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.46, "kelly": 0.373, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Zion Williamson", "name2": "Nikola Joki\u0107", "line1": 17.5, "line2": 27.5, "prediction1": 22.72, "prediction2": 22.11, "side1": "over", "side2": "under", "recommendation": 1, "ev": 7.15, "kelly": 0.357, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Chaz Lanier", "name2": "Nikola Joki\u0107", "line1": 5.5, "line2": 27.5, "prediction1": 8.23, "prediction2": 22.11, "side1": "over", "side2": "under", "recommendation": 0, "ev": 6.22, "kelly": 0.311, "sigma1": "Med", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Chaz Lanier", "name2": "Josh Giddey", "line1": 5.5, "line2": 18.5, "prediction1": 8.23, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.07, "kelly": 0.303, "sigma1": "Med", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Chaz Lanier", "name2": "Zion Williamson", "line1": 5.5, "line2": 17.5, "prediction1": 8.23, "prediction2": 22.72, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.52, "kelly": 0.276, "sigma1": "Med", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Karl-Anthony Towns", "name2": "D'Angelo Russell", "line1": 28.5, "line2": 11.5, "prediction1": 24.27, "prediction2": 15.64, "side1": "under", "side2": "over", "recommendation": 0, "ev": 3.51, "kelly": 0.175, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Karl-Anthony Towns", "name2": "Alex Caruso", "line1": 28.5, "line2": 5.5, "prediction1": 24.27, "prediction2": 7.32, "side1": "under", "side2": "over", "recommendation": 0, "ev": 3.48, "kelly": 0.174, "sigma1": "High", "sigma2": "Med", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "D'Angelo Russell", "name2": "Jaylin Williams", "line1": 11.5, "line2": 5.5, "prediction1": 15.64, "prediction2": 7.3, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.42, "kelly": 0.171, "sigma1": "High", "sigma2": "Med", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Karl-Anthony Towns", "name2": "Jaylin Williams", "line1": 28.5, "line2": 5.5, "prediction1": 24.27, "prediction2": 7.3, "side1": "under", "side2": "over", "recommendation": 0, "ev": 3.35, "kelly": 0.168, "sigma1": "High", "sigma2": "Med", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
];const underdogTriosData = [
    {"name1": "Zion Williamson", "name2": "Nikola Joki\u0107", "name3": "Josh Giddey", "line1": 17.5, "line2": 27.5, "line3": 18.5, "prediction1": 22.72, "prediction2": 22.11, "prediction3": 25.38, "side1": "over", "side2": "under", "side3": "over", "recommendation": 1, "ev": 14.71, "kelly": 0.294, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Chaz Lanier", "name2": "Nikola Joki\u0107", "name3": "Josh Giddey", "line1": 5.5, "line2": 27.5, "line3": 18.5, "prediction1": 8.23, "prediction2": 22.11, "prediction3": 25.38, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 13.27, "kelly": 0.265, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Chaz Lanier", "name2": "Karl-Anthony Towns", "name3": "Zion Williamson", "line1": 5.5, "line2": 28.5, "line3": 17.5, "prediction1": 8.23, "prediction2": 24.27, "prediction3": 22.72, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 10.13, "kelly": 0.203, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Karl-Anthony Towns", "name2": "D'Angelo Russell", "name3": "Alex Caruso", "line1": 28.5, "line2": 11.5, "line3": 5.5, "prediction1": 24.27, "prediction2": 15.64, "prediction3": 7.32, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.93, "kelly": 0.139, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "D'Angelo Russell", "name2": "P.J. Washington", "name3": "Jaylin Williams", "line1": 11.5, "line2": 15.5, "line3": 5.5, "prediction1": 15.64, "prediction2": 19.53, "prediction3": 7.3, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.6, "kelly": 0.132, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 38.2, "l5_2": 0.2, "l15_2": 0.33, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "P.J. Washington", "name2": "Alex Caruso", "name3": "Jaylin Williams", "line1": 15.5, "line2": 5.5, "line3": 5.5, "prediction1": 19.53, "prediction2": 7.32, "prediction3": 7.3, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.47, "kelly": 0.129, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "hitRate1": 38.2, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Josh Hart", "name2": "Jose Alvarado", "name3": "Ajay Mitchell", "line1": 12.5, "line2": 7.5, "line3": 14.5, "prediction1": 9.48, "prediction2": 9.81, "prediction3": 17.43, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 4.86, "kelly": 0.097, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Jalen Duren", "name2": "Miles Bridges", "name3": "Jose Alvarado", "line1": 20.5, "line2": 19.5, "line3": 7.5, "prediction1": 17.16, "prediction2": 22.79, "prediction3": 9.81, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 4.76, "kelly": 0.095, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Jalen Duren", "name2": "Josh Hart", "name3": "Ajay Mitchell", "line1": 20.5, "line2": 12.5, "line3": 14.5, "prediction1": 17.16, "prediction2": 9.48, "prediction3": 17.43, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 4.53, "kelly": 0.091, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Tony Bradley", "name2": "Miles Bridges", "name3": "Andrew Wiggins", "line1": 5.5, "line2": 19.5, "line3": 18.5, "prediction1": 6.45, "prediction2": 22.79, "prediction3": 21.08, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 3.72, "kelly": 0.074, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
];// This is a large data file - I'll create a simplified version that includes all the hit rates data
// For brevity, I'll include a condensed version with the key structures
const prizepicksPointsHitRates = [
    {"name": "Trey Murphy III", "line": 16.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.933, "underPct": 0.067},
    {"name": "Saddiq Bey", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.882, "underPct": 0.118},
    {"name": "Zion Williamson", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.861, "underPct": 0.139},
    {"name": "Naji Marshall", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.86, "underPct": 0.14},
    {"name": "Josh Giddey", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.85, "underPct": 0.15},
    {"name": "Bennedict Mathurin", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.835, "underPct": 0.165},
    {"name": "Jose Alvarado", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.819, "underPct": 0.181},
    {"name": "Kon Knueppel", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.815, "underPct": 0.185},
    {"name": "Isaac Okoro", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.811, "underPct": 0.189},
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.767, "underPct": 0.233},
    {"name": "Jaden McDaniels", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.744, "underPct": 0.256},
    {"name": "Sandro Mamukelashvili", "line": 8.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.727, "underPct": 0.273},
    {"name": "Collin Murray-Boyles", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.719, "underPct": 0.281},
    {"name": "Tony Bradley", "line": 4.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.718, "underPct": 0.282},
    {"name": "Cade Cunningham", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.716, "underPct": 0.284},
    {"name": "Tre Mann", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.713, "underPct": 0.287},
    {"name": "Kyle Kuzma", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.711, "underPct": 0.289},
    {"name": "Gradey Dick", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.704, "underPct": 0.296},
    {"name": "Simone Fontecchio", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.699, "underPct": 0.301},
    {"name": "LaMelo Ball", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.694, "underPct": 0.306},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.688, "underPct": 0.312},
    {"name": "Julius Randle", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.687, "underPct": 0.313},
    {"name": "Miles Bridges", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.683, "underPct": 0.317},
    {"name": "Jeremiah Fears", "line": 13.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.682, "underPct": 0.318},
    {"name": "Pelle Larsson", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.681, "underPct": 0.319},
    {"name": "Ryan Rollins", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.675, "underPct": 0.325},
    {"name": "Myles Turner", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.674, "underPct": 0.326},
    {"name": "Jarrett Allen", "line": 14.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.658, "underPct": 0.342},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.655, "underPct": 0.345},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.655, "underPct": 0.345},
    {"name": "Mike Conley", "line": 5.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.638, "underPct": 0.362},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.636, "underPct": 0.364},
    {"name": "Bobby Portis", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.628, "underPct": 0.372},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.625, "underPct": 0.375},
    {"name": "Ayo Dosunmu", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.615, "underPct": 0.385},
    {"name": "Cooper Flagg", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.612, "underPct": 0.388},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.61, "underPct": 0.39},
    {"name": "Patrick Williams", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.609, "underPct": 0.391},
    {"name": "Ajay Mitchell", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.601, "underPct": 0.399},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.571, "underPct": 0.429},
    {"name": "James Harden", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.568, "underPct": 0.432},
    {"name": "Pascal Siakam", "line": 23.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.565, "underPct": 0.435},
    {"name": "Brook Lopez", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.561, "underPct": 0.439},
    {"name": "Alex Caruso", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.559, "underPct": 0.441},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.555, "underPct": 0.445},
    {"name": "Giannis Antetokounmpo", "line": 31.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.55, "underPct": 0.45},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.544, "underPct": 0.456},
    {"name": "Aaron Gordon", "line": 18.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.537, "underPct": 0.463},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.525, "underPct": 0.475},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.524, "underPct": 0.476},
    {"name": "Sam Merrill", "line": 11.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.515, "underPct": 0.485},
    {"name": "De'Andre Hunter", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Andrew Nembhard", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.499, "underPct": 0.501},
    {"name": "Duncan Robinson", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Cole Anthony", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.492, "underPct": 0.508},
    {"name": "Tyrese Maxey", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.481, "underPct": 0.519},
    {"name": "Derik Queen", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.477, "underPct": 0.523},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.472, "underPct": 0.528},
    {"name": "Andrew Wiggins", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.47, "underPct": 0.53},
    {"name": "Kevin Huerter", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.466, "underPct": 0.534},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.452, "underPct": 0.548},
    {"name": "Lonzo Ball", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.45, "underPct": 0.55},
    {"name": "T.J. McConnell", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.449, "underPct": 0.551},
    {"name": "P.J. Washington", "line": 16.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.448, "underPct": 0.552},
    {"name": "Collin Sexton", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.439, "underPct": 0.561},
    {"name": "Quentin Grimes", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.43, "underPct": 0.57},
    {"name": "Jarace Walker", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.422, "underPct": 0.578},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.402, "underPct": 0.598},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Naz Reid", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.394, "underPct": 0.606},
    {"name": "Kris Dunn", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.387, "underPct": 0.613},
    {"name": "Landry Shamet", "line": 13.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.377, "underPct": 0.623},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.366, "underPct": 0.634},
    {"name": "Kel'el Ware", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.366, "underPct": 0.634},
    {"name": "D'Angelo Russell", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.361, "underPct": 0.639},
    {"name": "Matas Buzelis", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.354, "underPct": 0.646},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.321, "underPct": 0.679},
    {"name": "Nicolas Batum", "line": 5.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.306, "underPct": 0.694},
    {"name": "Jamal Murray", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.287, "underPct": 0.713},
    {"name": "Karl-Anthony Towns", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.285, "underPct": 0.715},
    {"name": "Andre Drummond", "line": 11.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.281, "underPct": 0.719},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.274, "underPct": 0.726},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.246, "underPct": 0.754},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.24, "underPct": 0.76},
    {"name": "Javonte Green", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.22, "underPct": 0.78},
    {"name": "John Collins", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.216, "underPct": 0.784},
    {"name": "Cameron Johnson", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.176, "underPct": 0.824},
    {"name": "Mitchell Robinson", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.161, "underPct": 0.839},
    {"name": "Jordan Clarkson", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.154, "underPct": 0.846},
    {"name": "Peyton Watson", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.104, "underPct": 0.896},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.075, "underPct": 0.925},
    {"name": "Kobe Sanders", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.064, "underPct": 0.936},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.026, "underPct": 0.974},
];const prizepicksAssistsHitRates = [
    {"name": "Josh Giddey", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.777, "underPct": 0.223},
    {"name": "LaMelo Ball", "line": 7.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.762, "underPct": 0.238},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.695, "underPct": 0.305},
    {"name": "Isaac Okoro", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.666, "underPct": 0.334},
    {"name": "Alex Caruso", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.651, "underPct": 0.349},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.578, "underPct": 0.422},
    {"name": "Lonzo Ball", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.568, "underPct": 0.432},
    {"name": "Pascal Siakam", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.558, "underPct": 0.442},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.549, "underPct": 0.451},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.522, "underPct": 0.478},
    {"name": "Jaylin Williams", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 4.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.495, "underPct": 0.505},
    {"name": "James Harden", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.484, "underPct": 0.516},
    {"name": "Jamal Murray", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.477, "underPct": 0.523},
    {"name": "Andrew Nembhard", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.464, "underPct": 0.536},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.462, "underPct": 0.538},
    {"name": "Javonte Green", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.448, "underPct": 0.552},
    {"name": "Giannis Antetokounmpo", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.445, "underPct": 0.555},
    {"name": "Ryan Rollins", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.437, "underPct": 0.563},
    {"name": "Shai Gilgeous-Alexander", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.433, "underPct": 0.567},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.424, "underPct": 0.576},
    {"name": "Evan Mobley", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.406, "underPct": 0.594},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.343, "underPct": 0.657},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.338, "underPct": 0.662},
    {"name": "Tyrese Maxey", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.334, "underPct": 0.666},
    {"name": "Josh Hart", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.231, "underPct": 0.769},
    {"name": "Miles McBride", "line": 4.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.174, "underPct": 0.826},
];const prizepicksReboundsHitRates = [
    {"name": "Trendon Watford", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.807, "underPct": 0.193},
    {"name": "Josh Giddey", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.795, "underPct": 0.205},
    {"name": "Matas Buzelis", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.786, "underPct": 0.214},
    {"name": "LaMelo Ball", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.718, "underPct": 0.282},
    {"name": "Donovan Mitchell", "line": 4.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.712, "underPct": 0.288},
    {"name": "Kon Knueppel", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.697, "underPct": 0.303},
    {"name": "James Harden", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.697, "underPct": 0.303},
    {"name": "Zion Williamson", "line": 5.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.689, "underPct": 0.311},
    {"name": "Mitchell Robinson", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.689, "underPct": 0.311},
    {"name": "Luguentz Dort", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.684, "underPct": 0.316},
    {"name": "Jalen Smith", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.683, "underPct": 0.317},
    {"name": "Saddiq Bey", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.671, "underPct": 0.329},
    {"name": "Lonzo Ball", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.638, "underPct": 0.362},
    {"name": "Scottie Barnes", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.623, "underPct": 0.377},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.618, "underPct": 0.382},
    {"name": "Ben Sheppard", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.616, "underPct": 0.384},
    {"name": "Collin Murray-Boyles", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.615, "underPct": 0.385},
    {"name": "Cade Cunningham", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.598, "underPct": 0.402},
    {"name": "Isaiah Hartenstein", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.581, "underPct": 0.419},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.581, "underPct": 0.419},
    {"name": "Brandon Ingram", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.566, "underPct": 0.434},
    {"name": "Bennedict Mathurin", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.547, "underPct": 0.453},
    {"name": "Javonte Green", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.533, "underPct": 0.467},
    {"name": "VJ Edgecombe", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.524, "underPct": 0.476},
    {"name": "Jarrett Allen", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.512, "underPct": 0.488},
    {"name": "Isaiah Jackson", "line": 6.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.511, "underPct": 0.489},
    {"name": "Jarace Walker", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.51, "underPct": 0.49},
    {"name": "Kel'el Ware", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.507, "underPct": 0.493},
    {"name": "Bobby Portis", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.506, "underPct": 0.494},
    {"name": "Ryan Rollins", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.504, "underPct": 0.496},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.502, "underPct": 0.498},
    {"name": "Ivica Zubac", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.494, "underPct": 0.506},
    {"name": "De'Andre Hunter", "line": 4.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.474, "underPct": 0.526},
    {"name": "P.J. Washington", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.457, "underPct": 0.543},
    {"name": "Evan Mobley", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.44, "underPct": 0.56},
    {"name": "Peyton Watson", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.414, "underPct": 0.586},
    {"name": "Cooper Flagg", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.406, "underPct": 0.594},
    {"name": "Dean Wade", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.405, "underPct": 0.595},
    {"name": "Aaron Gordon", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.404, "underPct": 0.596},
    {"name": "Ausar Thompson", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.402, "underPct": 0.598},
    {"name": "Miles Bridges", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.398, "underPct": 0.602},
    {"name": "Derik Queen", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.375, "underPct": 0.625},
    {"name": "Jakob Poeltl", "line": 8.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.373, "underPct": 0.627},
    {"name": "Ryan Kalkbrenner", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.368, "underPct": 0.632},
    {"name": "Josh Hart", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.352, "underPct": 0.648},
    {"name": "Bruce Brown", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.343, "underPct": 0.657},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.332, "underPct": 0.668},
    {"name": "Chet Holmgren", "line": 8.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.322, "underPct": 0.678},
    {"name": "Giannis Antetokounmpo", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.317, "underPct": 0.683},
    {"name": "Andrew Wiggins", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.296, "underPct": 0.704},
    {"name": "Anthony Edwards", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.294, "underPct": 0.706},
    {"name": "Jordan Clarkson", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.286, "underPct": 0.714},
    {"name": "Rudy Gobert", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.265, "underPct": 0.735},
    {"name": "Kobe Sanders", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.259, "underPct": 0.741},
    {"name": "Brook Lopez", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.233, "underPct": 0.767},
    {"name": "John Collins", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.209, "underPct": 0.791},
    {"name": "Andre Drummond", "line": 12.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.157, "underPct": 0.843},
];const prizepicksBlocksHitRates = [
    {"name": "Ausar Thompson", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Nicolas Batum", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.389, "underPct": 0.611},
    {"name": "Jaylin Williams", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.547, "underPct": 0.453},
];const prizepicksStealsHitRates = [
    {"name": "Ausar Thompson", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.52, "underPct": 0.48},
    {"name": "Bennedict Mathurin", "line": 0.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.283, "underPct": 0.717},
    {"name": "Duncan Robinson", "line": 0.5, "l5": 1.0, "l10": 0.5, "l15": 0.4, "overPct": 0.574, "underPct": 0.426},
    {"name": "Jarace Walker", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.455, "underPct": 0.545},
    {"name": "Jeremiah Robinson-Earl", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.457, "underPct": 0.543},
    {"name": "Ivica Zubac", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.485, "underPct": 0.515},
    {"name": "Nicolas Batum", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.473, "underPct": 0.527},
    {"name": "Dominick Barlow", "line": 0.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.12, "underPct": 0.88},
    {"name": "Kobe Sanders", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.447, "underPct": 0.553},
    {"name": "Cole Anthony", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.537, "underPct": 0.463},
    {"name": "Dean Wade", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.533, "underPct": 0.467},
    {"name": "Kyle Kuzma", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Sam Merrill", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.439, "underPct": 0.561},
    {"name": "Miles Bridges", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.39, "underPct": 0.61},
    {"name": "Sion James", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.545, "underPct": 0.455},
    {"name": "Tre Mann", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.457, "underPct": 0.543},
    {"name": "Collin Sexton", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.494, "underPct": 0.506},
    {"name": "Max Christie", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.483, "underPct": 0.517},
    {"name": "Mike Conley", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.51, "underPct": 0.49},
    {"name": "Rudy Gobert", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.436, "underPct": 0.564},
    {"name": "Shai Gilgeous-Alexander", "line": 1.5, "l5": 1.0, "l10": 0.5, "l15": 0.47, "overPct": 0.363, "underPct": 0.637},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Naji Marshall", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 16.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 41.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Nembhard", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jakob Poeltl", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 27.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Miles Bridges", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "LaMelo Ball", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Merrill", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 39.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pelle Larsson", "line": 16.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 34.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alex Caruso", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kevin Huerter", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 41.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nicolas Batum", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaac Okoro", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Patrick Williams", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jose Alvarado", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylin Williams", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Murray-Boyles", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Simone Fontecchio", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tre Mann", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 33.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derik Queen", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cole Anthony", "line": 15.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Javonte Green", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Jackson", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dominick Barlow", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Giannis Antetokounmpo", "line": 49.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 19.0, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sion James", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jarace Walker", "line": 16.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Landry Shamet", "line": 17.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarrett Allen", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jeremiah Robinson-Earl", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "VJ Edgecombe", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Scottie Barnes", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 44.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mike Conley", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 29.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Clarkson", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Miles McBride", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "T.J. McConnell", "line": 15.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Max Christie", "line": 14.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Sandro Mamukelashvili", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 22.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cade Cunningham", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Myles Turner", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaac Okoro", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "LaMelo Ball", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Rollins", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 23.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tony Bradley", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Smith", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Huerter", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Caruso", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Williams", "line": 16.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Merrill", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Kalkbrenner", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Mann", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Murray-Boyles", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 24.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Simone Fontecchio", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Isaiah Hartenstein", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Jackson", "line": 15.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Javonte Green", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Giannis Antetokounmpo", "line": 42.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 17.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nicolas Batum", "line": 9.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Dunn", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mitchell Robinson", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Norman Powell", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dean Wade", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cole Anthony", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sion James", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Robinson-Earl", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "T.J. McConnell", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ben Sheppard", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 23.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Miles McBride", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Landry Shamet", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 7.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cameron Johnson", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPAHitRates = [
    {"name": "Cade Cunningham", "line": 36.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 11.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brook Lopez", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandon Williams", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Zion Williamson", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Norman Powell", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Merrill", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Saddiq Bey", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Huerter", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Smith", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Caruso", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Hartenstein", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Gordon", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nicolas Batum", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Giannis Antetokounmpo", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Murray-Boyles", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Wiggins", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tre Mann", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kobe Sanders", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Duren", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Dunn", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 37.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Jackson", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lonzo Ball", "line": 12.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cole Anthony", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sion James", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "D'Angelo Russell", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "P.J. Washington", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles McBride", "line": 17.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jarace Walker", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Landry Shamet", "line": 14.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Karl-Anthony Towns", "line": 31.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bruce Brown", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donte DiVincenzo", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "T.J. McConnell", "line": 14.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksRAHitRates = [
    {"name": "LaMelo Ball", "line": 12.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 5.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Scottie Barnes", "line": 13.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 10.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 16.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Williams", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Mann", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sam Merrill", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Huerter", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kon Knueppel", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trendon Watford", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bennedict Mathurin", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "VJ Edgecombe", "line": 8.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 9.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Evan Mobley", "line": 13.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Giannis Antetokounmpo", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 11.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 6.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gradey Dick", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jakob Poeltl", "line": 11.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jeremiah Robinson-Earl", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mikal Bridges", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pelle Larsson", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Davion Mitchell", "line": 11.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 16.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 10.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 15.0, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cameron Johnson", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles McBride", "line": 7.0, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const prizepicksTurnoversHitRates = [
    {"name": "Jarrett Allen", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ben Sheppard", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Caruso", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tony Bradley", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Mitchell", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jose Alvarado", "line": 1.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Aaron Gordon", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Naz Reid", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 1.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Saddiq Bey", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Huerter", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Simone Fontecchio", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Chet Holmgren", "line": 2.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
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

