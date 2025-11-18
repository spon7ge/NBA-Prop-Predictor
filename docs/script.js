const prizepicksSinglesData = [
    {"name": "Josh Giddey", "bookmaker": "BetRivers", "line": 18.5, "prediction": 25.38, "side": "Over", "odds": 105, "recommendation": 1, "ev": 6.57, "roi": 65.7, "kelly": 0.626, "sigma": "High"},
    {"name": "Bennedict Mathurin", "bookmaker": "FanDuel", "line": 16.5, "prediction": 20.83, "side": "Over", "odds": 102, "recommendation": 0, "ev": 6.23, "roi": 62.3, "kelly": 0.611, "sigma": "Med"},
    {"name": "Zion Williamson", "bookmaker": "BetRivers", "line": 18.5, "prediction": 22.72, "side": "Over", "odds": 117, "recommendation": 0, "ev": 6.08, "roi": 60.8, "kelly": 0.52, "sigma": "High"},
    {"name": "Josh Giddey", "bookmaker": "FanDuel", "line": 17.5, "prediction": 25.38, "side": "Over", "odds": -111, "recommendation": 1, "ev": 5.77, "roi": 57.7, "kelly": 0.64, "sigma": "High"},
    {"name": "Josh Giddey", "bookmaker": "DraftKings", "line": 17.5, "prediction": 25.38, "side": "Over", "odds": -115, "recommendation": 1, "ev": 5.75, "roi": 57.5, "kelly": 0.661, "sigma": "High"},
    {"name": "Nikola Jokic", "bookmaker": "DraftKings", "line": 27.5, "prediction": 22.09, "side": "Under", "odds": -106, "recommendation": 1, "ev": 5.67, "roi": 56.7, "kelly": 0.601, "sigma": "High"},
    {"name": "Nikola Jokic", "bookmaker": "BetRivers", "line": 26.5, "prediction": 22.09, "side": "Under", "odds": 107, "recommendation": 0, "ev": 5.66, "roi": 56.6, "kelly": 0.529, "sigma": "High"},
    {"name": "Bennedict Mathurin", "bookmaker": "DraftKings", "line": 15.5, "prediction": 20.83, "side": "Over", "odds": -121, "recommendation": 1, "ev": 5.56, "roi": 55.6, "kelly": 0.673, "sigma": "Med"},
    {"name": "Josh Giddey", "bookmaker": "BetRivers", "line": 17.5, "prediction": 25.38, "side": "Over", "odds": -117, "recommendation": 1, "ev": 5.48, "roi": 54.8, "kelly": 0.641, "sigma": "High"},
    {"name": "Nikola Jokic", "bookmaker": "BetMGM", "line": 27.5, "prediction": 22.09, "side": "Under", "odds": -110, "recommendation": 1, "ev": 5.4, "roi": 54.0, "kelly": 0.594, "sigma": "High"},
    {"name": "Bennedict Mathurin", "bookmaker": "BetMGM", "line": 16.5, "prediction": 20.83, "side": "Over", "odds": -110, "recommendation": 0, "ev": 5.3, "roi": 53.0, "kelly": 0.583, "sigma": "Med"},
    {"name": "Nikola Jokic", "bookmaker": "BetRivers", "line": 27.5, "prediction": 22.09, "side": "Under", "odds": -112, "recommendation": 1, "ev": 5.22, "roi": 52.2, "kelly": 0.585, "sigma": "High"},
    {"name": "LaMelo Ball", "bookmaker": "DraftKings", "line": 19.5, "prediction": 25.58, "side": "Over", "odds": -110, "recommendation": 1, "ev": 5.19, "roi": 51.9, "kelly": 0.571, "sigma": "High"},
    {"name": "Zion Williamson", "bookmaker": "BetRivers", "line": 17.5, "prediction": 22.72, "side": "Over", "odds": -107, "recommendation": 1, "ev": 5.18, "roi": 51.8, "kelly": 0.555, "sigma": "High"},
    {"name": "Bennedict Mathurin", "bookmaker": "BetRivers", "line": 16.5, "prediction": 20.83, "side": "Over", "odds": -115, "recommendation": 0, "ev": 5.18, "roi": 51.8, "kelly": 0.596, "sigma": "Med"},
];const prizepicksPairsData = [
    {"name1": "Nikola Joki\u0107", "name2": "Josh Giddey", "line1": 0.5, "line2": 17.5, "prediction1": 22.09, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 1, "ev": 12.56, "kelly": 0.628, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 85.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "LaMelo Ball", "name2": "Nikola Joki\u0107", "line1": 19.5, "line2": 0.5, "prediction1": 25.58, "prediction2": 22.09, "side1": "over", "side2": "over", "recommendation": 1, "ev": 11.73, "kelly": 0.587, "sigma1": "High", "sigma2": "High", "hitRate1": 69.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Simone Fontecchio", "name2": "Nikola Joki\u0107", "line1": 8.5, "line2": 0.5, "prediction1": 13.35, "prediction2": 22.09, "side1": "over", "side2": "over", "recommendation": 1, "ev": 11.55, "kelly": 0.578, "sigma1": "High", "sigma2": "High", "hitRate1": 69.9, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Bennedict Mathurin", "name2": "Josh Giddey", "line1": 16.5, "line2": 17.5, "prediction1": 20.83, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 0, "ev": 8.18, "kelly": 0.409, "sigma1": "Med", "sigma2": "High", "hitRate1": 83.5, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 85.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "LaMelo Ball", "name2": "Josh Giddey", "line1": 19.5, "line2": 17.5, "prediction1": 25.58, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.98, "kelly": 0.399, "sigma1": "High", "sigma2": "High", "hitRate1": 69.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 85.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Bennedict Mathurin", "name2": "Simone Fontecchio", "line1": 16.5, "line2": 8.5, "prediction1": 20.83, "prediction2": 13.35, "side1": "over", "side2": "over", "recommendation": 0, "ev": 7.39, "kelly": 0.37, "sigma1": "Med", "sigma2": "High", "hitRate1": 83.5, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 69.9, "l5_2": 0.8, "l15_2": 0.73},
    {"name1": "LaMelo Ball", "name2": "Simone Fontecchio", "line1": 19.5, "line2": 8.5, "prediction1": 25.58, "prediction2": 13.35, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.2, "kelly": 0.36, "sigma1": "High", "sigma2": "High", "hitRate1": 69.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 69.9, "l5_2": 0.8, "l15_2": 0.73},
    {"name1": "Bennedict Mathurin", "name2": "Naji Marshall", "line1": 16.5, "line2": 10.0, "prediction1": 20.83, "prediction2": 15.16, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.74, "kelly": 0.337, "sigma1": "Med", "sigma2": "High", "hitRate1": 83.5, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 86.0, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Ryan Kalkbrenner", "name2": "Naji Marshall", "line1": 8.5, "line2": 10.0, "prediction1": 11.86, "prediction2": 15.16, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.62, "kelly": 0.281, "sigma1": "Med", "sigma2": "High", "hitRate1": 62.5, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 86.0, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Jarace Walker", "name2": "Naji Marshall", "line1": 9.5, "line2": 10.0, "prediction1": 13.72, "prediction2": 15.16, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.41, "kelly": 0.271, "sigma1": "High", "sigma2": "High", "hitRate1": 42.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 86.0, "l5_2": 0.6, "l15_2": 0.33},
];const prizepicksTriosData = [
    {"name1": "Bennedict Mathurin", "name2": "Nikola Joki\u0107", "name3": "Josh Giddey", "line1": 16.5, "line2": 0.5, "line3": 17.5, "prediction1": 20.83, "prediction2": 22.09, "prediction3": 25.38, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 22.56, "kelly": 0.451, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 83.5, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 85.0, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Simone Fontecchio", "name2": "Nikola Joki\u0107", "name3": "Josh Giddey", "line1": 8.5, "line2": 0.5, "line3": 17.5, "prediction1": 13.35, "prediction2": 22.09, "prediction3": 25.38, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 22.52, "kelly": 0.45, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 69.9, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 85.0, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Bennedict Mathurin", "name2": "LaMelo Ball", "name3": "Simone Fontecchio", "line1": 16.5, "line2": 19.5, "line3": 8.5, "prediction1": 20.83, "prediction2": 25.58, "prediction3": 13.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 14.76, "kelly": 0.295, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 83.5, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 69.4, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 69.9, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "LaMelo Ball", "name2": "Cooper Flagg", "name3": "Naji Marshall", "line1": 19.5, "line2": 15.5, "line3": 10.0, "prediction1": 25.58, "prediction2": 20.98, "prediction3": 15.16, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 13.02, "kelly": 0.26, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 69.4, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 61.2, "l5_2": 0.8, "l15_2": 0.53, "hitRate3": 86.0, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "Ryan Kalkbrenner", "name2": "Cooper Flagg", "name3": "Naji Marshall", "line1": 8.5, "line2": 15.5, "line3": 10.0, "prediction1": 11.86, "prediction2": 20.98, "prediction3": 15.16, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.47, "kelly": 0.229, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 62.5, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 61.2, "l5_2": 0.8, "l15_2": 0.53, "hitRate3": 86.0, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "Jarace Walker", "name2": "Ryan Kalkbrenner", "name3": "Brandon Williams", "line1": 9.5, "line2": 8.5, "line3": 13.5, "prediction1": 13.72, "prediction2": 11.86, "prediction3": 18.22, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.39, "kelly": 0.188, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 42.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 62.5, "l5_2": 1.0, "l15_2": 0.6, "hitRate3": 36.6, "l5_3": 0.8, "l15_3": 0.33},
    {"name1": "Jarace Walker", "name2": "Tony Bradley", "name3": "Brandon Williams", "line1": 9.5, "line2": 4.5, "line3": 13.5, "prediction1": 13.72, "prediction2": 6.24, "prediction3": 18.22, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.61, "kelly": 0.172, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "hitRate1": 42.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 71.8, "l5_2": 1.0, "l15_2": 0.6, "hitRate3": 36.6, "l5_3": 0.8, "l15_3": 0.33},
    {"name1": "Tony Bradley", "name2": "Brook Lopez", "name3": "D'Angelo Russell", "line1": 4.5, "line2": 6.5, "line3": 11.5, "prediction1": 6.24, "prediction2": 9.01, "prediction3": 15.64, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.33, "kelly": 0.147, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "hitRate1": 71.8, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 56.1, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 36.1, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Kris Dunn", "name2": "Brook Lopez", "name3": "D'Angelo Russell", "line1": 7.5, "line2": 6.5, "line3": 11.5, "prediction1": 10.19, "prediction2": 9.01, "prediction3": 15.64, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.09, "kelly": 0.142, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 38.7, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 56.1, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 36.1, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Kris Dunn", "name2": "Collin Murray-Boyles", "name3": "Jose Alvarado", "line1": 7.5, "line2": 6.5, "line3": 7.5, "prediction1": 10.19, "prediction2": 8.52, "prediction3": 9.81, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.95, "kelly": 0.119, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 38.7, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 71.9, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 81.9, "l5_3": 0.8, "l15_3": 0.4},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Nikola Joki\u0107", "name2": "Josh Giddey", "line1": 27.5, "line2": 18.5, "prediction1": 22.09, "prediction2": 25.38, "side1": "under", "side2": "over", "recommendation": 1, "ev": 7.36, "kelly": 0.368, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 79.1, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Ryan Kalkbrenner", "name2": "Nikola Joki\u0107", "line1": 8.5, "line2": 27.5, "prediction1": 11.86, "prediction2": 22.09, "side1": "over", "side2": "under", "recommendation": 0, "ev": 6.34, "kelly": 0.317, "sigma1": "Med", "sigma2": "High", "hitRate1": 62.5, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Ryan Kalkbrenner", "name2": "Josh Giddey", "line1": 8.5, "line2": 18.5, "prediction1": 11.86, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.06, "kelly": 0.303, "sigma1": "Med", "sigma2": "High", "hitRate1": 62.5, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 79.1, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "D'Angelo Russell", "name2": "Nikola Joki\u0107", "line1": 11.5, "line2": 27.5, "prediction1": 15.64, "prediction2": 22.09, "side1": "over", "side2": "under", "recommendation": 0, "ev": 5.64, "kelly": 0.282, "sigma1": "High", "sigma2": "High", "hitRate1": 36.1, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Chaz Lanier", "name2": "Josh Giddey", "line1": 5.5, "line2": 18.5, "prediction1": 7.35, "prediction2": 25.38, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.28, "kelly": 0.264, "sigma1": "Med", "sigma2": "High", "hitRate1": 20.7, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 79.1, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Ryan Kalkbrenner", "name2": "D'Angelo Russell", "line1": 8.5, "line2": 11.5, "prediction1": 11.86, "prediction2": 15.64, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.19, "kelly": 0.21, "sigma1": "Med", "sigma2": "High", "hitRate1": 62.5, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 36.1, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Karl-Anthony Towns", "name2": "D'Angelo Russell", "line1": 28.5, "line2": 11.5, "prediction1": 24.56, "prediction2": 15.64, "side1": "under", "side2": "over", "recommendation": 0, "ev": 3.32, "kelly": 0.166, "sigma1": "High", "sigma2": "High", "hitRate1": 77.6, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 36.1, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Chaz Lanier", "name2": "P.J. Washington", "line1": 5.5, "line2": 15.5, "prediction1": 7.35, "prediction2": 19.53, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.29, "kelly": 0.164, "sigma1": "Med", "sigma2": "High", "hitRate1": 20.7, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 54.7, "l5_2": 0.4, "l15_2": 0.4},
    {"name1": "Karl-Anthony Towns", "name2": "P.J. Washington", "line1": 28.5, "line2": 15.5, "prediction1": 24.56, "prediction2": 19.53, "side1": "under", "side2": "over", "recommendation": 0, "ev": 3.15, "kelly": 0.157, "sigma1": "High", "sigma2": "High", "hitRate1": 77.6, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 54.7, "l5_2": 0.4, "l15_2": 0.4},
    {"name1": "Chaz Lanier", "name2": "Karl-Anthony Towns", "line1": 5.5, "line2": 28.5, "prediction1": 7.35, "prediction2": 24.56, "side1": "over", "side2": "under", "recommendation": 0, "ev": 3.01, "kelly": 0.15, "sigma1": "Med", "sigma2": "High", "hitRate1": 20.7, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 77.6, "l5_2": 0.2, "l15_2": 0.13},
];const underdogTriosData = [
    {"name1": "Ryan Kalkbrenner", "name2": "Nikola Joki\u0107", "name3": "Josh Giddey", "line1": 8.5, "line2": 27.5, "line3": 18.5, "prediction1": 11.86, "prediction2": 22.09, "prediction3": 25.38, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 13.25, "kelly": 0.265, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 62.5, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 79.1, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "D'Angelo Russell", "name2": "Nikola Joki\u0107", "name3": "Josh Giddey", "line1": 11.5, "line2": 27.5, "line3": 18.5, "prediction1": 15.64, "prediction2": 22.09, "prediction3": 25.38, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 12.2, "kelly": 0.244, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 36.1, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 79.1, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Ryan Kalkbrenner", "name2": "Karl-Anthony Towns", "name3": "D'Angelo Russell", "line1": 8.5, "line2": 28.5, "line3": 11.5, "prediction1": 11.86, "prediction2": 24.56, "prediction3": 15.64, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 7.7, "kelly": 0.154, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 62.5, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 77.6, "l5_2": 0.2, "l15_2": 0.13, "hitRate3": 36.1, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Chaz Lanier", "name2": "Karl-Anthony Towns", "name3": "P.J. Washington", "line1": 5.5, "line2": 28.5, "line3": 15.5, "prediction1": 7.35, "prediction2": 24.56, "prediction3": 19.53, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 6.32, "kelly": 0.126, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 20.7, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 77.6, "l5_2": 0.2, "l15_2": 0.13, "hitRate3": 54.7, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Chaz Lanier", "name2": "P.J. Washington", "name3": "Jose Alvarado", "line1": 5.5, "line2": 15.5, "line3": 7.5, "prediction1": 7.35, "prediction2": 19.53, "prediction3": 9.81, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.04, "kelly": 0.121, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 20.7, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 54.7, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 81.9, "l5_3": 0.8, "l15_3": 0.4},
    {"name1": "Miles Bridges", "name2": "Andrew Wiggins", "name3": "Jose Alvarado", "line1": 19.5, "line2": 18.5, "line3": 7.5, "prediction1": 22.98, "prediction2": 21.08, "prediction3": 9.81, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 4.46, "kelly": 0.089, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 75.6, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 47.0, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 81.9, "l5_3": 0.8, "l15_3": 0.4},
    {"name1": "Miles Bridges", "name2": "Andrew Wiggins", "name3": "Josh Hart", "line1": 19.5, "line2": 18.5, "line3": 12.5, "prediction1": 22.98, "prediction2": 21.08, "prediction3": 9.77, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 3.64, "kelly": 0.073, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 75.6, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 47.0, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 59.8, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Josh Hart", "name2": "Pelle Larsson", "name3": "Alex Caruso", "line1": 12.5, "line2": 9.5, "line3": 5.5, "prediction1": 9.77, "prediction2": 11.28, "prediction3": 6.44, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 2.99, "kelly": 0.06, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "hitRate1": 59.8, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 68.1, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 55.9, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Pelle Larsson", "name2": "Alex Caruso", "name3": "Jaylin Williams", "line1": 9.5, "line2": 5.5, "line3": 5.5, "prediction1": 11.28, "prediction2": 6.44, "prediction3": 6.43, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 2.85, "kelly": 0.057, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "hitRate1": 68.1, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 55.9, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 55.2, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "James Harden", "name2": "Kyle Kuzma", "name3": "Jaylin Williams", "line1": 25.5, "line2": 12.5, "line3": 5.5, "prediction1": 27.79, "prediction2": 14.85, "prediction3": 6.43, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 2.54, "kelly": 0.051, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 64.3, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 71.1, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 55.2, "l5_3": 0.6, "l15_3": 0.4},
];// This is a large data file - I'll create a simplified version that includes all the hit rates data
// For brevity, I'll include a condensed version with the key structures
const prizepicksPointsHitRates = [
    {"name": "Trey Murphy III", "line": 16.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.933, "underPct": 0.067},
    {"name": "Saddiq Bey", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.882, "underPct": 0.118},
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
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.655, "underPct": 0.345},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.655, "underPct": 0.345},
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
    {"name": "Josh Hart", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.402, "underPct": 0.598},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
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
    {"name": "Julius Randle", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.581, "underPct": 0.419},
    {"name": "Isaiah Hartenstein", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.581, "underPct": 0.419},
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
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cade Cunningham", "line": 41.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Nembhard", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 27.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jakob Poeltl", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Merrill", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 39.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pelle Larsson", "line": 16.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Saddiq Bey", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 34.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Miles Bridges", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Huerter", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Caruso", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Nicolas Batum", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 33.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Dunn", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaac Okoro", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Patrick Williams", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jose Alvarado", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Joe", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Murray-Boyles", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Simone Fontecchio", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Mann", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 41.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylin Williams", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 19.0, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lonzo Ball", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Javonte Green", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Duncan Robinson", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Jackson", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Giannis Antetokounmpo", "line": 49.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cole Anthony", "line": 15.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sion James", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Robinson-Earl", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Landry Shamet", "line": 17.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarrett Allen", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 16.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Scottie Barnes", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 29.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mike Conley", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Karl-Anthony Towns", "line": 44.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jordan Clarkson", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "T.J. McConnell", "line": 15.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Sandro Mamukelashvili", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 14.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Trey Murphy III", "line": 22.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jakob Poeltl", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Myles Turner", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaac Okoro", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sam Merrill", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 23.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andre Drummond", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Duncan Robinson", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tony Bradley", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Simone Fontecchio", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Smith", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Huerter", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 16.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Hartenstein", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Caruso", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Mann", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Murray-Boyles", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Wiggins", "line": 24.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ayo Dosunmu", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ausar Thompson", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Jackson", "line": 15.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Javonte Green", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Nicolas Batum", "line": 9.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyle Kuzma", "line": 17.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Giannis Antetokounmpo", "line": 42.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Naz Reid", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mitchell Robinson", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dean Wade", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cole Anthony", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sion James", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Robinson-Earl", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 23.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "T.J. McConnell", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Miles McBride", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Landry Shamet", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mike Conley", "line": 7.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cameron Johnson", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPAHitRates = [
    {"name": "Cade Cunningham", "line": 36.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 11.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brook Lopez", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Shead", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandon Williams", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sam Merrill", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
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
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Murray-Boyles", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kel'el Ware", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Mann", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jarrett Allen", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kris Dunn", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trendon Watford", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kobe Sanders", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyrese Maxey", "line": 37.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Jackson", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lonzo Ball", "line": 12.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cole Anthony", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Hart", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ayo Dosunmu", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sion James", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "P.J. Washington", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Landry Shamet", "line": 14.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles McBride", "line": 17.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jordan Clarkson", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Karl-Anthony Towns", "line": 31.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bruce Brown", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donte DiVincenzo", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "T.J. McConnell", "line": 14.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksRAHitRates = [
    {"name": "LaMelo Ball", "line": 12.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 5.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Derik Queen", "line": 10.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Scottie Barnes", "line": 13.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 16.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Williams", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Merrill", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tre Mann", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Huerter", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trendon Watford", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 11.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bennedict Mathurin", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "VJ Edgecombe", "line": 8.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 9.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Evan Mobley", "line": 13.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Giannis Antetokounmpo", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 6.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mitchell Robinson", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
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
    {"name": "Trey Murphy III", "line": 16.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.933, "underPct": 0.067},
    {"name": "Jose Alvarado", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.819, "underPct": 0.181},
    {"name": "Kon Knueppel", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.815, "underPct": 0.185},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.791, "underPct": 0.209},
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.767, "underPct": 0.233},
    {"name": "Miles Bridges", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.756, "underPct": 0.244},
    {"name": "Kyle Kuzma", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.711, "underPct": 0.289},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.688, "underPct": 0.312},
    {"name": "Pelle Larsson", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.681, "underPct": 0.319},
    {"name": "Myles Turner", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.674, "underPct": 0.326},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.655, "underPct": 0.345},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.655, "underPct": 0.345},
    {"name": "James Harden", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.643, "underPct": 0.357},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.636, "underPct": 0.364},
    {"name": "Aaron Gordon", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.628, "underPct": 0.372},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.625, "underPct": 0.375},
    {"name": "Ayo Dosunmu", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.615, "underPct": 0.385},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.61, "underPct": 0.39},
    {"name": "Patrick Williams", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.609, "underPct": 0.391},
    {"name": "Ajay Mitchell", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.601, "underPct": 0.399},
    {"name": "Lonzo Ball", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.598, "underPct": 0.402},
    {"name": "Alex Caruso", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.559, "underPct": 0.441},
    {"name": "Tony Bradley", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.558, "underPct": 0.442},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.555, "underPct": 0.445},
    {"name": "Jaylin Williams", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.552, "underPct": 0.448},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.547, "underPct": 0.453},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.524, "underPct": 0.476},
    {"name": "De'Andre Hunter", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Duncan Robinson", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Tyrese Maxey", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.481, "underPct": 0.519},
    {"name": "Derik Queen", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.477, "underPct": 0.523},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.472, "underPct": 0.528},
    {"name": "Andrew Wiggins", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.47, "underPct": 0.53},
    {"name": "Kevin Huerter", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.466, "underPct": 0.534},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.452, "underPct": 0.548},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.416, "underPct": 0.584},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.402, "underPct": 0.598},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Naz Reid", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.394, "underPct": 0.606},
    {"name": "D'Angelo Russell", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.361, "underPct": 0.639},
    {"name": "Matas Buzelis", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.354, "underPct": 0.646},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.321, "underPct": 0.679},
    {"name": "Jamal Murray", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.287, "underPct": 0.713},
    {"name": "Caris LeVert", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.271, "underPct": 0.729},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.246, "underPct": 0.754},
    {"name": "Karl-Anthony Towns", "line": 28.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.224, "underPct": 0.776},
    {"name": "Javonte Green", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.22, "underPct": 0.78},
    {"name": "John Collins", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.216, "underPct": 0.784},
    {"name": "Chaz Lanier", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.207, "underPct": 0.793},
    {"name": "Peyton Watson", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.183, "underPct": 0.817},
    {"name": "Cameron Johnson", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.176, "underPct": 0.824},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.075, "underPct": 0.925},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.026, "underPct": 0.974},
    {"name": "Daniss Jenkins", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.02, "underPct": 0.98},
];const underdogAssistsHitRates = [
    {"name": "Josh Giddey", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.777, "underPct": 0.223},
    {"name": "Isaac Okoro", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.666, "underPct": 0.334},
    {"name": "Alex Caruso", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.651, "underPct": 0.349},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.578, "underPct": 0.422},
    {"name": "Lonzo Ball", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.568, "underPct": 0.432},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.549, "underPct": 0.451},
    {"name": "Jaylin Williams", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.484, "underPct": 0.516},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.462, "underPct": 0.538},
    {"name": "Javonte Green", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.448, "underPct": 0.552},
    {"name": "Caris LeVert", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.43, "underPct": 0.57},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.391, "underPct": 0.609},
    {"name": "Jalen Duren", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.323, "underPct": 0.677},
    {"name": "Chaz Lanier", "line": 1.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.269, "underPct": 0.731},
    {"name": "Daniss Jenkins", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.059, "underPct": 0.941},
];const underdogReboundsHitRates = [
    {"name": "Matas Buzelis", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.786, "underPct": 0.214},
    {"name": "Donovan Mitchell", "line": 4.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.712, "underPct": 0.288},
    {"name": "Mitchell Robinson", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.689, "underPct": 0.311},
    {"name": "Luguentz Dort", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.684, "underPct": 0.316},
    {"name": "Jalen Smith", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.683, "underPct": 0.317},
    {"name": "Ben Sheppard", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.616, "underPct": 0.384},
    {"name": "Collin Murray-Boyles", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.615, "underPct": 0.385},
    {"name": "Isaac Okoro", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.605, "underPct": 0.395},
    {"name": "Jalen Duren", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.583, "underPct": 0.417},
    {"name": "Isaiah Hartenstein", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.581, "underPct": 0.419},
    {"name": "Evan Mobley", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.572, "underPct": 0.428},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.539, "underPct": 0.461},
    {"name": "Jarrett Allen", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.512, "underPct": 0.488},
    {"name": "Jarace Walker", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.51, "underPct": 0.49},
    {"name": "Kel'el Ware", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.507, "underPct": 0.493},
    {"name": "Ryan Rollins", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.504, "underPct": 0.496},
    {"name": "Duncan Robinson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.499, "underPct": 0.501},
    {"name": "De'Andre Hunter", "line": 4.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.474, "underPct": 0.526},
    {"name": "P.J. Washington", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.457, "underPct": 0.543},
    {"name": "Dean Wade", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.405, "underPct": 0.595},
    {"name": "Anthony Edwards", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.294, "underPct": 0.706},
    {"name": "Jordan Clarkson", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.286, "underPct": 0.714},
    {"name": "Brook Lopez", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.233, "underPct": 0.767},
    {"name": "Caris LeVert", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.177, "underPct": 0.823},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
    {"name": "Daniss Jenkins", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.438, "underPct": 0.562},
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.476, "underPct": 0.524},
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.378, "underPct": 0.622},
    {"name": "Shai Gilgeous-Alexander", "line": 1.5, "l5": 1.0, "l10": 0.5, "l15": 0.47, "overPct": 0.363, "underPct": 0.637},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Sandro Mamukelashvili", "line": 13.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Duren", "line": 35.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 39.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alex Caruso", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Josh Giddey", "line": 34.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Saddiq Bey", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sam Merrill", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pelle Larsson", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bobby Portis", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kevin Huerter", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaac Okoro", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylin Williams", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Gordon", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Kalkbrenner", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kel'el Ware", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniss Jenkins", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "James Harden", "line": 40.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Simone Fontecchio", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Murray-Boyles", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jose Alvarado", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kon Knueppel", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Mann", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Caris LeVert", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Chaz Lanier", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brook Lopez", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Giannis Antetokounmpo", "line": 49.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Javonte Green", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 18.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lonzo Ball", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cole Anthony", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Davion Mitchell", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sion James", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 16.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "John Collins", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jeremiah Robinson-Earl", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarrett Allen", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Miles McBride", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Karl-Anthony Towns", "line": 44.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jordan Clarkson", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bruce Brown", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogPRHitRates = [
    {"name": "Trey Murphy III", "line": 22.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 32.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kon Knueppel", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Wiggins", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniss Jenkins", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Josh Giddey", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kel'el Ware", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Giannis Antetokounmpo", "line": 42.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jarrett Allen", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const underdogPAHitRates = [
    {"name": "Trey Murphy III", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shai Gilgeous-Alexander", "line": 37.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Wiggins", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Giannis Antetokounmpo", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Daniss Jenkins", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const underdogRAHitRates = [
    {"name": "LaMelo Ball", "line": 12.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Daniss Jenkins", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lonzo Ball", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Robinson-Earl", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogTurnoversHitRates = [
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
];const underdogBlocksStealsHitRates = [
    {"name": "Ryan Kalkbrenner", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Rudy Gobert", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Chet Holmgren", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (volatile), Med, Low (consistent)</div>
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
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (volatile), Med, Low (consistent)</div>
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

