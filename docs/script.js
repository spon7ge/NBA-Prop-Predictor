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
    {"name1": "KJ Simpson", "name2": "Jake LaRavia", "line1": 11.5, "line2": 4.5, "odds1": -120, "odds2": -122, "prediction1": 3.09, "prediction2": 9.42, "modelProb1": 0.912, "modelProb2": 0.906, "side1": "under", "side2": "over", "parlayProb": 0.826, "parlayOdds": 234, "evPercent": 176.02, "kellyQuarter": 0.1881},
    {"name1": "Ja'Kobe Walter", "name2": "Buddy Hield", "line1": 7.5, "line2": 12.5, "odds1": 104, "odds2": -110, "prediction1": 2.91, "prediction2": 5.4, "modelProb1": 0.765, "modelProb2": 0.824, "side1": "under", "side2": "under", "parlayProb": 0.63, "parlayOdds": 289, "evPercent": 144.99, "kellyQuarter": 0.1254},
    {"name1": "Tidjane Sala\u00fcn", "name2": "Quinten Post", "line1": 6.5, "line2": 10.5, "odds1": -105, "odds2": -113, "prediction1": 2.17, "prediction2": 3.86, "modelProb1": 0.789, "modelProb2": 0.816, "side1": "under", "side2": "under", "parlayProb": 0.644, "parlayOdds": 268, "evPercent": 137.12, "kellyQuarter": 0.1279},
    {"name1": "Brandon Miller", "name2": "Kenrich Williams", "line1": 21.5, "line2": 7.5, "odds1": -106, "odds2": -122, "prediction1": 15.33, "prediction2": 2.54, "modelProb1": 0.753, "modelProb2": 0.837, "side1": "under", "side2": "under", "parlayProb": 0.63, "parlayOdds": 254, "evPercent": 123.15, "kellyQuarter": 0.1212},
    {"name1": "VJ Edgecombe", "name2": "Jalen Williams", "line1": 10.0, "line2": 23.5, "odds1": -137, "odds2": -105, "prediction1": 14.51, "prediction2": 16.7, "modelProb1": 0.837, "modelProb2": 0.746, "side1": "over", "side2": "under", "parlayProb": 0.624, "parlayOdds": 238, "evPercent": 110.89, "kellyQuarter": 0.1165},
    {"name1": "Cameron Johnson", "name2": "Ace Bailey", "line1": 13.5, "line2": 12.5, "odds1": 100, "odds2": -115, "prediction1": 8.83, "prediction2": 6.91, "modelProb1": 0.719, "modelProb2": 0.772, "side1": "under", "side2": "under", "parlayProb": 0.555, "parlayOdds": 274, "evPercent": 107.51, "kellyQuarter": 0.0981},
    {"name1": "Anfernee Simons", "name2": "Kyle Filipowski", "line1": 11.5, "line2": 10.5, "odds1": 100, "odds2": -107, "prediction1": 14.38, "prediction2": 6.19, "modelProb1": 0.717, "modelProb2": 0.7, "side1": "over", "side2": "under", "parlayProb": 0.502, "parlayOdds": 287, "evPercent": 94.29, "kellyQuarter": 0.0821},
    {"name1": "Deni Avdija", "name2": "Deandre Ayton", "line1": 25.5, "line2": 14.5, "odds1": -115, "odds2": 104, "prediction1": 20.3, "prediction2": 16.37, "modelProb1": 0.711, "modelProb2": 0.651, "side1": "under", "side2": "over", "parlayProb": 0.463, "parlayOdds": 281, "evPercent": 76.51, "kellyQuarter": 0.0681},
    {"name1": "Jordan Walsh", "name2": "Jimmy Butler III", "line1": 7.5, "line2": 21.5, "odds1": -119, "odds2": -113, "prediction1": 3.88, "prediction2": 16.52, "modelProb1": 0.717, "modelProb2": 0.693, "side1": "under", "side2": "under", "parlayProb": 0.497, "parlayOdds": 247, "evPercent": 72.38, "kellyQuarter": 0.0733},
    {"name1": "Nikola Joki\u0107", "name2": "Jaylin Williams", "line1": 29.5, "line2": 8.0, "odds1": -115, "odds2": -137, "prediction1": 23.79, "prediction2": 3.86, "modelProb1": 0.693, "modelProb2": 0.739, "side1": "under", "side2": "under", "parlayProb": 0.512, "parlayOdds": 223, "evPercent": 65.23, "kellyQuarter": 0.0731},
];const prizepicksTriosData = [
    {"name1": "KJ Simpson", "name2": "Buddy Hield", "name3": "Jake LaRavia", "line1": 11.5, "line2": 12.5, "line3": 4.5, "odds1": -120, "odds2": -110, "odds3": -122, "prediction1": 3.09, "prediction2": 5.4, "prediction3": 9.42, "modelProb1": 0.912, "modelProb2": 0.824, "modelProb3": 0.906, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.681, "parlayOdds": 537, "evPercent": 333.65, "kellyQuarter": 0.1553},
    {"name1": "Ja'Kobe Walter", "name2": "Tidjane Sala\u00fcn", "name3": "Quinten Post", "line1": 7.5, "line2": 6.5, "line3": 10.5, "odds1": 104, "odds2": -105, "odds3": -113, "prediction1": 2.91, "prediction2": 2.17, "prediction3": 3.86, "modelProb1": 0.765, "modelProb2": 0.789, "modelProb3": 0.816, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.493, "parlayOdds": 651, "evPercent": 269.96, "kellyQuarter": 0.1037},
    {"name1": "Brandon Miller", "name2": "VJ Edgecombe", "name3": "Kenrich Williams", "line1": 21.5, "line2": 10.0, "line3": 7.5, "odds1": -106, "odds2": -137, "odds3": -122, "prediction1": 15.33, "prediction2": 14.51, "prediction3": 2.54, "modelProb1": 0.753, "modelProb2": 0.837, "modelProb3": 0.837, "side1": "under", "side2": "over", "side3": "under", "parlayProb": 0.528, "parlayOdds": 512, "evPercent": 222.86, "kellyQuarter": 0.1088},
    {"name1": "Anfernee Simons", "name2": "Cameron Johnson", "name3": "Jalen Williams", "line1": 11.5, "line2": 13.5, "line3": 23.5, "odds1": 100, "odds2": 100, "odds3": -105, "prediction1": 14.38, "prediction2": 8.83, "prediction3": 16.7, "modelProb1": 0.717, "modelProb2": 0.719, "modelProb3": 0.746, "side1": "over", "side2": "under", "side3": "under", "parlayProb": 0.385, "parlayOdds": 681, "evPercent": 200.39, "kellyQuarter": 0.0736},
    {"name1": "Deni Avdija", "name2": "Deandre Ayton", "name3": "Ace Bailey", "line1": 25.5, "line2": 14.5, "line3": 12.5, "odds1": -115, "odds2": 104, "odds3": -115, "prediction1": 20.3, "prediction2": 16.37, "prediction3": 6.91, "modelProb1": 0.711, "modelProb2": 0.651, "modelProb3": 0.772, "side1": "under", "side2": "over", "side3": "under", "parlayProb": 0.357, "parlayOdds": 613, "evPercent": 154.86, "kellyQuarter": 0.0632},
    {"name1": "Jordan Walsh", "name2": "Jimmy Butler III", "name3": "Kyle Filipowski", "line1": 7.5, "line2": 21.5, "line3": 10.5, "odds1": -119, "odds2": -113, "odds3": -107, "prediction1": 3.88, "prediction2": 16.52, "prediction3": 6.19, "modelProb1": 0.717, "modelProb2": 0.693, "modelProb3": 0.7, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.348, "parlayOdds": 571, "evPercent": 133.27, "kellyQuarter": 0.0583},
    {"name1": "Nikola Joki\u0107", "name2": "Santi Aldama", "name3": "Jaylin Williams", "line1": 29.5, "line2": 11.5, "line3": 8.0, "odds1": -115, "odds2": -105, "odds3": -137, "prediction1": 23.79, "prediction2": 13.0, "prediction3": 3.86, "modelProb1": 0.693, "modelProb2": 0.646, "modelProb3": 0.739, "side1": "under", "side2": "over", "side3": "under", "parlayProb": 0.331, "parlayOdds": 531, "evPercent": 108.59, "kellyQuarter": 0.0511},
    {"name1": "Peyton Watson", "name2": "Brandin Podziemski", "name3": "Keyonte George", "line1": 12.5, "line2": 9.5, "line3": 21.5, "odds1": -112, "odds2": -104, "odds3": -107, "prediction1": 8.53, "prediction2": 10.57, "prediction3": 17.69, "modelProb1": 0.681, "modelProb2": 0.618, "modelProb3": 0.647, "side1": "under", "side2": "over", "side3": "under", "parlayProb": 0.272, "parlayOdds": 618, "evPercent": 95.48, "kellyQuarter": 0.0386},
    {"name1": "Cam Spencer", "name2": "Tyrese Maxey", "name3": "Aaron Wiggins", "line1": 11.5, "line2": 27.5, "line3": 15.5, "odds1": 100, "odds2": 105, "odds3": -104, "prediction1": 8.49, "prediction2": 28.28, "prediction3": 17.63, "modelProb1": 0.605, "modelProb2": 0.58, "modelProb3": 0.634, "side1": "under", "side2": "over", "side3": "over", "parlayProb": 0.222, "parlayOdds": 704, "evPercent": 78.79, "kellyQuarter": 0.028},
    {"name1": "Sion James", "name2": "Rui Hachimura", "name3": "Lauri Markkanen", "line1": 6.5, "line2": 11.5, "line3": 23.5, "odds1": -105, "odds2": 105, "odds3": -116, "prediction1": 3.89, "prediction2": 11.9, "prediction3": 25.62, "modelProb1": 0.602, "modelProb2": 0.565, "modelProb3": 0.646, "side1": "under", "side2": "over", "side3": "over", "parlayProb": 0.22, "parlayOdds": 645, "evPercent": 63.97, "kellyQuarter": 0.0248},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "KJ Simpson", "name2": "VJ Edgecombe", "line1": 11.5, "line2": 10.5, "odds1": -120, "odds2": 110, "prediction1": 3.09, "prediction2": 14.51, "modelProb1": 0.912, "modelProb2": 0.806, "side1": "under", "side2": "over", "parlayProb": 0.735, "parlayOdds": 285, "evPercent": 183.16, "kellyQuarter": 0.1607},
    {"name1": "Spencer Jones", "name2": "Buddy Hield", "line1": 6.5, "line2": 12.5, "odds1": -137, "odds2": -110, "prediction1": 1.21, "prediction2": 5.4, "modelProb1": 0.923, "modelProb2": 0.824, "side1": "under", "side2": "under", "parlayProb": 0.76, "parlayOdds": 230, "evPercent": 150.8, "kellyQuarter": 0.1639},
    {"name1": "Ja'Kobe Walter", "name2": "Quinten Post", "line1": 7.5, "line2": 10.5, "odds1": 104, "odds2": -113, "prediction1": 2.91, "prediction2": 3.86, "modelProb1": 0.765, "modelProb2": 0.816, "side1": "under", "side2": "under", "parlayProb": 0.624, "parlayOdds": 285, "evPercent": 140.3, "kellyQuarter": 0.1231},
    {"name1": "Brandon Miller", "name2": "Jalen Williams", "line1": 21.5, "line2": 23.5, "odds1": -106, "odds2": -105, "prediction1": 15.33, "prediction2": 16.7, "modelProb1": 0.753, "modelProb2": 0.746, "side1": "under", "side2": "under", "parlayProb": 0.562, "parlayOdds": 279, "evPercent": 112.83, "kellyQuarter": 0.1011},
    {"name1": "Cameron Johnson", "name2": "Ace Bailey", "line1": 13.5, "line2": 12.5, "odds1": 100, "odds2": -115, "prediction1": 8.83, "prediction2": 6.91, "modelProb1": 0.719, "modelProb2": 0.772, "side1": "under", "side2": "under", "parlayProb": 0.555, "parlayOdds": 274, "evPercent": 107.51, "kellyQuarter": 0.0981},
    {"name1": "Jordan Walsh", "name2": "Deni Avdija", "line1": 7.5, "line2": 25.5, "odds1": -119, "odds2": -115, "prediction1": 3.88, "prediction2": 20.3, "modelProb1": 0.717, "modelProb2": 0.711, "side1": "under", "side2": "under", "parlayProb": 0.51, "parlayOdds": 244, "evPercent": 75.35, "kellyQuarter": 0.0772},
    {"name1": "Nikola Joki\u0107", "name2": "Jimmy Butler III", "line1": 29.5, "line2": 21.5, "odds1": -115, "odds2": -113, "prediction1": 23.79, "prediction2": 16.52, "modelProb1": 0.693, "modelProb2": 0.693, "side1": "under", "side2": "under", "parlayProb": 0.48, "parlayOdds": 252, "evPercent": 69.0, "kellyQuarter": 0.0684},
    {"name1": "Peyton Watson", "name2": "Santi Aldama", "line1": 12.5, "line2": 11.5, "odds1": -112, "odds2": -105, "prediction1": 8.53, "prediction2": 13.0, "modelProb1": 0.681, "modelProb2": 0.646, "side1": "under", "side2": "over", "parlayProb": 0.44, "parlayOdds": 270, "evPercent": 62.75, "kellyQuarter": 0.0581},
    {"name1": "Brandin Podziemski", "name2": "Keyonte George", "line1": 9.5, "line2": 21.5, "odds1": -104, "odds2": -107, "prediction1": 10.57, "prediction2": 17.69, "modelProb1": 0.618, "modelProb2": 0.647, "side1": "over", "side2": "under", "parlayProb": 0.4, "parlayOdds": 279, "evPercent": 51.59, "kellyQuarter": 0.0462},
    {"name1": "Cam Spencer", "name2": "Aaron Wiggins", "line1": 11.5, "line2": 15.5, "odds1": 100, "odds2": -104, "prediction1": 8.49, "prediction2": 17.63, "modelProb1": 0.605, "modelProb2": 0.634, "side1": "under", "side2": "over", "parlayProb": 0.384, "parlayOdds": 292, "evPercent": 50.36, "kellyQuarter": 0.0431},
];const underdogTriosData = [
    {"name1": "KJ Simpson", "name2": "Buddy Hield", "name3": "VJ Edgecombe", "line1": 11.5, "line2": 12.5, "line3": 10.5, "odds1": -120, "odds2": -110, "odds3": 110, "prediction1": 3.09, "prediction2": 5.4, "prediction3": 14.51, "modelProb1": 0.912, "modelProb2": 0.824, "modelProb3": 0.806, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.606, "parlayOdds": 635, "evPercent": 345.32, "kellyQuarter": 0.136},
    {"name1": "Ja'Kobe Walter", "name2": "Spencer Jones", "name3": "Quinten Post", "line1": 7.5, "line2": 6.5, "line3": 10.5, "odds1": 104, "odds2": -137, "odds3": -113, "prediction1": 2.91, "prediction2": 1.21, "prediction3": 3.86, "modelProb1": 0.765, "modelProb2": 0.923, "modelProb3": 0.816, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.576, "parlayOdds": 565, "evPercent": 282.92, "kellyQuarter": 0.1252},
    {"name1": "Brandon Miller", "name2": "Deni Avdija", "name3": "Jalen Williams", "line1": 21.5, "line2": 25.5, "line3": 23.5, "odds1": -106, "odds2": -115, "odds3": -105, "prediction1": 15.33, "prediction2": 20.3, "prediction3": 16.7, "modelProb1": 0.753, "modelProb2": 0.711, "modelProb3": 0.746, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.399, "parlayOdds": 609, "evPercent": 183.22, "kellyQuarter": 0.0752},
    {"name1": "Jordan Walsh", "name2": "Cameron Johnson", "name3": "Ace Bailey", "line1": 7.5, "line2": 13.5, "line3": 12.5, "odds1": -119, "odds2": 100, "odds3": -115, "prediction1": 3.88, "prediction2": 8.83, "prediction3": 6.91, "modelProb1": 0.717, "modelProb2": 0.719, "modelProb3": 0.772, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.398, "parlayOdds": 588, "evPercent": 173.54, "kellyQuarter": 0.0738},
    {"name1": "Nikola Joki\u0107", "name2": "Santi Aldama", "name3": "Jimmy Butler III", "line1": 29.5, "line2": 11.5, "line3": 21.5, "odds1": -115, "odds2": -105, "odds3": -113, "prediction1": 23.79, "prediction2": 13.0, "prediction3": 16.52, "modelProb1": 0.693, "modelProb2": 0.646, "modelProb3": 0.693, "side1": "under", "side2": "over", "side3": "under", "parlayProb": 0.31, "parlayOdds": 588, "evPercent": 113.45, "kellyQuarter": 0.0482},
    {"name1": "Peyton Watson", "name2": "Brandin Podziemski", "name3": "Keyonte George", "line1": 12.5, "line2": 9.5, "line3": 21.5, "odds1": -112, "odds2": -104, "odds3": -107, "prediction1": 8.53, "prediction2": 10.57, "prediction3": 17.69, "modelProb1": 0.681, "modelProb2": 0.618, "modelProb3": 0.647, "side1": "under", "side2": "over", "side3": "under", "parlayProb": 0.272, "parlayOdds": 618, "evPercent": 95.48, "kellyQuarter": 0.0386},
    {"name1": "Cam Spencer", "name2": "Tyrese Maxey", "name3": "Aaron Wiggins", "line1": 11.5, "line2": 27.5, "line3": 15.5, "odds1": 100, "odds2": 105, "odds3": -104, "prediction1": 8.49, "prediction2": 28.28, "prediction3": 17.63, "modelProb1": 0.605, "modelProb2": 0.58, "modelProb3": 0.634, "side1": "under", "side2": "over", "side3": "over", "parlayProb": 0.222, "parlayOdds": 704, "evPercent": 78.79, "kellyQuarter": 0.028},
    {"name1": "Josh Giddey", "name2": "Paul George", "name3": "Ajay Mitchell", "line1": 20.5, "line2": 13.5, "line3": 18.5, "odds1": -120, "odds2": 102, "odds3": -115, "prediction1": 16.98, "prediction2": 10.89, "prediction3": 13.66, "modelProb1": 0.638, "modelProb2": 0.575, "modelProb3": 0.657, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.241, "parlayOdds": 592, "evPercent": 66.67, "kellyQuarter": 0.0282},
    {"name1": "Will Richard", "name2": "Rui Hachimura", "name3": "Lauri Markkanen", "line1": 7.5, "line2": 11.5, "line3": 23.5, "odds1": -118, "odds2": 105, "odds3": -116, "prediction1": 4.61, "prediction2": 11.9, "prediction3": 25.62, "modelProb1": 0.626, "modelProb2": 0.565, "modelProb3": 0.646, "side1": "under", "side2": "over", "side3": "over", "parlayProb": 0.229, "parlayOdds": 605, "evPercent": 61.21, "kellyQuarter": 0.0253},
    {"name1": "Sam Hauser", "name2": "Donovan Clingan", "name3": "Quentin Grimes", "line1": 6.5, "line2": 9.5, "line3": 13.5, "odds1": -105, "odds2": 103, "odds3": -122, "prediction1": 7.06, "prediction2": 9.75, "prediction3": 14.81, "modelProb1": 0.59, "modelProb2": 0.566, "modelProb3": 0.618, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.206, "parlayOdds": 621, "evPercent": 48.84, "kellyQuarter": 0.0197},
];const prizepicksPointsHitRates = [
    {"name": "Austin Reaves", "line": 23.0, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.934, "underPct": 0.066},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.828, "underPct": 0.172},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.812, "underPct": 0.188},
    {"name": "Coby White", "line": 19.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.799, "underPct": 0.201},
    {"name": "Jordan Walsh", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.772, "underPct": 0.228},
    {"name": "Jaylen Brown", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.757, "underPct": 0.243},
    {"name": "Jake LaRavia", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.755, "underPct": 0.245},
    {"name": "Cam Spencer", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.752, "underPct": 0.248},
    {"name": "Joel Embiid", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.749, "underPct": 0.251},
    {"name": "VJ Edgecombe", "line": 10.0, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.748, "underPct": 0.252},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.735, "underPct": 0.265},
    {"name": "Sam Hauser", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.734, "underPct": 0.266},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.706, "underPct": 0.294},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.703, "underPct": 0.297},
    {"name": "Cameron Johnson", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.702, "underPct": 0.298},
    {"name": "Justin Edwards", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.676, "underPct": 0.324},
    {"name": "Gabe Vincent", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.669, "underPct": 0.331},
    {"name": "Quentin Grimes", "line": 14.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.662, "underPct": 0.338},
    {"name": "Keyonte George", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.654, "underPct": 0.346},
    {"name": "Tyrese Maxey", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.648, "underPct": 0.352},
    {"name": "Peyton Watson", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.643, "underPct": 0.357},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.63, "underPct": 0.37},
    {"name": "Svi Mykhailiuk", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.63, "underPct": 0.37},
    {"name": "Scottie Barnes", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.628, "underPct": 0.372},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.627, "underPct": 0.373},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.593, "underPct": 0.407},
    {"name": "Derrick White", "line": 18.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.573, "underPct": 0.427},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.571, "underPct": 0.429},
    {"name": "Anfernee Simons", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.568, "underPct": 0.432},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.524, "underPct": 0.476},
    {"name": "Neemias Queta", "line": 11.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.503, "underPct": 0.497},
    {"name": "Paul George", "line": 14.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.492, "underPct": 0.508},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.482, "underPct": 0.518},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.433, "underPct": 0.567},
    {"name": "Lauri Markkanen", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.428, "underPct": 0.572},
    {"name": "Ja'Kobe Walter", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.421, "underPct": 0.579},
    {"name": "Ryan Kalkbrenner", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.418, "underPct": 0.582},
    {"name": "Santi Aldama", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.416, "underPct": 0.584},
    {"name": "Kon Knueppel", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.415, "underPct": 0.585},
    {"name": "Sandro Mamukelashvili", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.395, "underPct": 0.605},
    {"name": "Shaedon Sharpe", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.368, "underPct": 0.632},
    {"name": "Sion James", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.361, "underPct": 0.639},
    {"name": "Kris Murray", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.345, "underPct": 0.655},
    {"name": "Brandon Ingram", "line": 23.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.334, "underPct": 0.666},
    {"name": "Jakob Poeltl", "line": 10.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.327, "underPct": 0.673},
    {"name": "Josh Giddey", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.325, "underPct": 0.675},
    {"name": "Jaylin Williams", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.316, "underPct": 0.684},
    {"name": "Draymond Green", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.313, "underPct": 0.687},
    {"name": "Miles Bridges", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.302, "underPct": 0.698},
    {"name": "Cedric Coward", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.296, "underPct": 0.704},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.284, "underPct": 0.716},
    {"name": "Will Richard", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.253, "underPct": 0.747},
    {"name": "Aaron Wiggins", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.191, "underPct": 0.809},
    {"name": "LeBron James", "line": 17.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.161, "underPct": 0.839},
    {"name": "Chet Holmgren", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.155, "underPct": 0.845},
    {"name": "Kenrich Williams", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.146, "underPct": 0.854},
    {"name": "Brandon Miller", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.111, "underPct": 0.889},
    {"name": "Ajay Mitchell", "line": 18.0, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.04, "underPct": 0.96},
    {"name": "Jalen Williams", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.026, "underPct": 0.974},
    {"name": "KJ Simpson", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.004, "underPct": 0.996},
];const prizepicksAssistsHitRates = [
    {"name": "Coby White", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.731, "underPct": 0.269},
    {"name": "Austin Reaves", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.708, "underPct": 0.292},
    {"name": "LeBron James", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.655, "underPct": 0.345},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.613, "underPct": 0.387},
    {"name": "Jaylen Brown", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.61, "underPct": 0.39},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.586, "underPct": 0.414},
    {"name": "Toumani Camara", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.57, "underPct": 0.43},
    {"name": "Payton Pritchard", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.55, "underPct": 0.45},
    {"name": "Sandro Mamukelashvili", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.543, "underPct": 0.457},
    {"name": "Cam Spencer", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.505, "underPct": 0.495},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.463, "underPct": 0.537},
    {"name": "Derrick White", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.449, "underPct": 0.551},
    {"name": "Anfernee Simons", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.422, "underPct": 0.578},
    {"name": "Tyrese Maxey", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.361, "underPct": 0.639},
    {"name": "Kon Knueppel", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.358, "underPct": 0.642},
    {"name": "Jalen Williams", "line": 6.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.339, "underPct": 0.661},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.32, "underPct": 0.68},
    {"name": "Josh Giddey", "line": 9.0, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.293, "underPct": 0.707},
    {"name": "Jaylin Williams", "line": 3.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.26, "underPct": 0.74},
    {"name": "Ajay Mitchell", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.244, "underPct": 0.756},
    {"name": "Rui Hachimura", "line": 0.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.182, "underPct": 0.818},
];const prizepicksReboundsHitRates = [
    {"name": "Austin Reaves", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.743, "underPct": 0.257},
    {"name": "VJ Edgecombe", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.694, "underPct": 0.306},
    {"name": "Matas Buzelis", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.659, "underPct": 0.341},
    {"name": "Sion James", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.631, "underPct": 0.369},
    {"name": "Lauri Markkanen", "line": 6.0, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.595, "underPct": 0.405},
    {"name": "Donovan Clingan", "line": 10.0, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.586, "underPct": 0.414},
    {"name": "Jaylen Brown", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.584, "underPct": 0.416},
    {"name": "Cedric Coward", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.563, "underPct": 0.437},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Spencer Jones", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.555, "underPct": 0.445},
    {"name": "Draymond Green", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.543, "underPct": 0.457},
    {"name": "Paul George", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.543, "underPct": 0.457},
    {"name": "Rui Hachimura", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.506, "underPct": 0.494},
    {"name": "Santi Aldama", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.504, "underPct": 0.496},
    {"name": "Jalen Williams", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.502, "underPct": 0.498},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 12.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.499, "underPct": 0.501},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.488, "underPct": 0.512},
    {"name": "Josh Minott", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.485, "underPct": 0.515},
    {"name": "Jordan Walsh", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.466, "underPct": 0.534},
    {"name": "Jock Landale", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.466, "underPct": 0.534},
    {"name": "Ja'Kobe Walter", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.462, "underPct": 0.538},
    {"name": "Scottie Barnes", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.458, "underPct": 0.542},
    {"name": "Kris Murray", "line": 4.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.447, "underPct": 0.553},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.441, "underPct": 0.559},
    {"name": "Kyle Filipowski", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.437, "underPct": 0.563},
    {"name": "Deandre Ayton", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.436, "underPct": 0.564},
    {"name": "Brandon Ingram", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.435, "underPct": 0.565},
    {"name": "Derrick White", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.432, "underPct": 0.568},
    {"name": "Sandro Mamukelashvili", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.419, "underPct": 0.581},
    {"name": "Jakob Poeltl", "line": 9.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.416, "underPct": 0.584},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.414, "underPct": 0.586},
    {"name": "Kon Knueppel", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.407, "underPct": 0.593},
    {"name": "Ryan Kalkbrenner", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.405, "underPct": 0.595},
    {"name": "Bruce Brown", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.401, "underPct": 0.599},
    {"name": "Shaedon Sharpe", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.401, "underPct": 0.599},
    {"name": "Jerami Grant", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.39, "underPct": 0.61},
    {"name": "Cameron Johnson", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.385, "underPct": 0.615},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.374, "underPct": 0.626},
    {"name": "Chet Holmgren", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.345, "underPct": 0.655},
    {"name": "Peyton Watson", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.337, "underPct": 0.663},
    {"name": "Immanuel Quickley", "line": 4.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.337, "underPct": 0.663},
    {"name": "Joel Embiid", "line": 6.0, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.294, "underPct": 0.706},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.285, "underPct": 0.715},
    {"name": "Jaylin Williams", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.256, "underPct": 0.744},
    {"name": "Aaron Wiggins", "line": 3.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.184, "underPct": 0.816},
    {"name": "LeBron James", "line": 6.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.142, "underPct": 0.858},
    {"name": "KJ Simpson", "line": 3.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.083, "underPct": 0.917},
];const prizepicksBlocksHitRates = [
    {"name": "Jordan Walsh", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.504, "underPct": 0.496},
    {"name": "Josh Minott", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.374, "underPct": 0.626},
    {"name": "Ryan Kalkbrenner", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.463, "underPct": 0.537},
    {"name": "Brandon Miller", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.687, "underPct": 0.313},
    {"name": "Jerami Grant", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.54, "underPct": 0.46},
    {"name": "Santi Aldama", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.565, "underPct": 0.435},
    {"name": "Toumani Camara", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.541, "underPct": 0.459},
    {"name": "Jaylin Williams", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.345, "underPct": 0.655},
];const prizepicksStealsHitRates = [
    {"name": "Brandon Ingram", "line": 0.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.514, "underPct": 0.486},
    {"name": "Neemias Queta", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.421, "underPct": 0.579},
    {"name": "Sam Hauser", "line": 0.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.521, "underPct": 0.479},
    {"name": "Sion James", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.487, "underPct": 0.513},
    {"name": "Kentavious Caldwell-Pope", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.357, "underPct": 0.643},
    {"name": "Rui Hachimura", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.404, "underPct": 0.596},
    {"name": "Jalen Williams", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.263, "underPct": 0.737},
    {"name": "Keyonte George", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.603, "underPct": 0.397},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Jaylen Brown", "line": 41.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 27.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Wells", "line": 17.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 41.0, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Gabe Vincent", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 32.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tyrese Maxey", "line": 37.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Spencer", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Liam McNeeley", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Joel Embiid", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 21.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach Edey", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jared McCain", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cameron Johnson", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 31.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Hauser", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Justin Edwards", "line": 6.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Paul George", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Chet Holmgren", "line": 31.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 28.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 21.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sion James", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anfernee Simons", "line": 17.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 21.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Santi Aldama", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 32.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Aaron Wiggins", "line": 22.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kenrich Williams", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cason Wallace", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylin Williams", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jakob Poeltl", "line": 22.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Miller", "line": 28.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ja'Kobe Walter", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 34.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "KJ Simpson", "line": 20.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Jaylen Wells", "line": 16.0, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Coby White", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 35.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Draymond Green", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 19.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 26.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Joel Embiid", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 23.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cameron Johnson", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 29.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jared McCain", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sam Hauser", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Spencer", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jock Landale", "line": 12.0, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Paul George", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Justin Edwards", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 14.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ja'Kobe Walter", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 28.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sion James", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Peyton Watson", "line": 19.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kentavious Caldwell-Pope", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shaedon Sharpe", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Santi Aldama", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylin Williams", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jakob Poeltl", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Wiggins", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "LeBron James", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Chet Holmgren", "line": 29.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Will Richard", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Miller", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kon Knueppel", "line": 24.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.0, "l10": 0.0, "l15": 0.2, "overPct": 0.0, "underPct": 1.0},
    {"name": "Brandon Ingram", "line": 29.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "KJ Simpson", "line": 15.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Kenrich Williams", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jalen Williams", "line": 28.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPAHitRates = [
    {"name": "Deni Avdija", "line": 33.0, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 23.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Wells", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gabe Vincent", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rui Hachimura", "line": 11.0, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "VJ Edgecombe", "line": 12.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Spencer", "line": 16.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Hauser", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jock Landale", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 27.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Scottie Barnes", "line": 26.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jared McCain", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Joel Embiid", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cameron Johnson", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paul George", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jakob Poeltl", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles Bridges", "line": 26.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Miller", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Wiggins", "line": 18.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ja'Kobe Walter", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylin Williams", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "KJ Simpson", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jalen Williams", "line": 29.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Ajay Mitchell", "line": 23.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const prizepicksRAHitRates = [
    {"name": "Sam Hauser", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Draymond Green", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 9.0, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jared McCain", "line": 4.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "VJ Edgecombe", "line": 7.0, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Gabe Vincent", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keyonte George", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cason Wallace", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ace Bailey", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Minott", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 9.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 12.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 10.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paul George", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Joel Embiid", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Peyton Watson", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 9.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Miller", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ajay Mitchell", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Aaron Wiggins", "line": 6.0, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const prizepicksTurnoversHitRates = [
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 2.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Deandre Ayton", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Miller", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Liam McNeeley", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 1.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Brandon Miller", "line": 1.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Joel Embiid", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Justin Edwards", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylin Williams", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gabe Vincent", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const underdogPointsHitRates = [
    {"name": "Austin Reaves", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.934, "underPct": 0.066},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.828, "underPct": 0.172},
    {"name": "Jared McCain", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.778, "underPct": 0.222},
    {"name": "Jordan Walsh", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.772, "underPct": 0.228},
    {"name": "Cam Spencer", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.752, "underPct": 0.248},
    {"name": "Quentin Grimes", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.751, "underPct": 0.249},
    {"name": "Joel Embiid", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.749, "underPct": 0.251},
    {"name": "VJ Edgecombe", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.748, "underPct": 0.252},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.735, "underPct": 0.265},
    {"name": "Sam Hauser", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.734, "underPct": 0.266},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.706, "underPct": 0.294},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.703, "underPct": 0.297},
    {"name": "Cameron Johnson", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.702, "underPct": 0.298},
    {"name": "Jaylen Brown", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.697, "underPct": 0.303},
    {"name": "Spencer Jones", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.681, "underPct": 0.319},
    {"name": "Justin Edwards", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.676, "underPct": 0.324},
    {"name": "Gabe Vincent", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.669, "underPct": 0.331},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.662, "underPct": 0.338},
    {"name": "Keyonte George", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.654, "underPct": 0.346},
    {"name": "Tyrese Maxey", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.648, "underPct": 0.352},
    {"name": "Peyton Watson", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.643, "underPct": 0.357},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.63, "underPct": 0.37},
    {"name": "Svi Mykhailiuk", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.63, "underPct": 0.37},
    {"name": "Scottie Barnes", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.628, "underPct": 0.372},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.627, "underPct": 0.373},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.62, "underPct": 0.38},
    {"name": "Paul George", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.597, "underPct": 0.403},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.593, "underPct": 0.407},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.571, "underPct": 0.429},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.524, "underPct": 0.476},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.482, "underPct": 0.518},
    {"name": "Lauri Markkanen", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.428, "underPct": 0.572},
    {"name": "Santi Aldama", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.416, "underPct": 0.584},
    {"name": "Kon Knueppel", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.415, "underPct": 0.585},
    {"name": "Sandro Mamukelashvili", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.395, "underPct": 0.605},
    {"name": "Shaedon Sharpe", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.368, "underPct": 0.632},
    {"name": "Brandon Ingram", "line": 23.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.334, "underPct": 0.666},
    {"name": "Josh Giddey", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.325, "underPct": 0.675},
    {"name": "Jaylin Williams", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.316, "underPct": 0.684},
    {"name": "Miles Bridges", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.302, "underPct": 0.698},
    {"name": "Cedric Coward", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.296, "underPct": 0.704},
    {"name": "Will Richard", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.253, "underPct": 0.747},
    {"name": "Aaron Wiggins", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.191, "underPct": 0.809},
    {"name": "LeBron James", "line": 17.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.161, "underPct": 0.839},
    {"name": "Brandon Miller", "line": 20.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.16, "underPct": 0.84},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.04, "underPct": 0.96},
    {"name": "Jalen Williams", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.026, "underPct": 0.974},
    {"name": "KJ Simpson", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.004, "underPct": 0.996},
];const underdogAssistsHitRates = [
    {"name": "Toumani Camara", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.57, "underPct": 0.43},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.463, "underPct": 0.537},
    {"name": "Anfernee Simons", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.422, "underPct": 0.578},
    {"name": "Tyrese Maxey", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.361, "underPct": 0.639},
];const underdogReboundsHitRates = [
    {"name": "Austin Reaves", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.743, "underPct": 0.257},
    {"name": "Matas Buzelis", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.659, "underPct": 0.341},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Spencer Jones", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.555, "underPct": 0.445},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.541, "underPct": 0.459},
    {"name": "Josh Minott", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.485, "underPct": 0.515},
    {"name": "Ja'Kobe Walter", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.462, "underPct": 0.538},
    {"name": "Ryan Kalkbrenner", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.405, "underPct": 0.595},
    {"name": "Bruce Brown", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.401, "underPct": 0.599},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.285, "underPct": 0.715},
    {"name": "Aaron Wiggins", "line": 3.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.184, "underPct": 0.816},
    {"name": "KJ Simpson", "line": 3.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.083, "underPct": 0.917},
];const underdogBlocksHitRates = [
    {"name": "Ryan Kalkbrenner", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.463, "underPct": 0.537},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.392, "underPct": 0.608},
];const underdogStealsHitRates = [
    {"name": "Jalen Williams", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.263, "underPct": 0.737},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Deni Avdija", "line": 41.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 41.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 27.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Spencer Jones", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 37.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 32.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jordan Walsh", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Spencer", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Kalkbrenner", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Joel Embiid", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Liam McNeeley", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Hauser", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 31.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jared McCain", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Minott", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cameron Johnson", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Paul George", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Matas Buzelis", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Chet Holmgren", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Murray", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 32.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sion James", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shaedon Sharpe", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Santi Aldama", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Miller", "line": 28.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylin Williams", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ajay Mitchell", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "KJ Simpson", "line": 20.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jalen Williams", "line": 35.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPRHitRates = [
    {"name": "Coby White", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jaylen Brown", "line": 35.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deandre Ayton", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Joel Embiid", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Giddey", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shaedon Sharpe", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "LeBron James", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kon Knueppel", "line": 24.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Miller", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Aaron Wiggins", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 29.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Williams", "line": 28.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.0, "l10": 0.0, "l15": 0.2, "overPct": 0.0, "underPct": 1.0},
];const underdogPAHitRates = [
    {"name": "Deni Avdija", "line": 32.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 24.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 27.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Joel Embiid", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Chet Holmgren", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Miller", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Williams", "line": 29.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const underdogRAHitRates = [
    {"name": "Jaylen Brown", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const underdogTurnoversHitRates = [
    {"name": "Draymond Green", "line": 2.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Coby White", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Miller", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
];const underdogBlocksStealsHitRates = [
    {"name": "Peyton Watson", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
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
            <th style="width: 9%">Odds</th>
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
            <td style="font-weight: 600; color: ${row.odds > 0 ? '#f87171' : '#34d399'};" title="American odds format. Positive odds show potential profit on $100 bet. Negative odds show amount needed to bet to win $100.">
                ${formatOdds(row.odds)}
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
            <th style="width: 12%">Player 1</th>
            <th style="width: 5%">Line</th>
            <th style="width: 5%">Proj.</th>
            <th style="width: 5%">Odds</th>
            <th style="width: 6%">Model Prob</th>
            <th style="width: 12%">Player 2</th>
            <th style="width: 5%">Line</th>
            <th style="width: 5%">Proj.</th>
            <th style="width: 5%">Odds</th>
            <th style="width: 6%">Model Prob</th>
            <th style="width: 6%">Parlay Prob</th>
            <th style="width: 6%">Parlay Odds</th>
            <th style="width: 6%">EV%</th>
            <th style="width: 6%">Kelly</th>
        </tr>
    `;

    const tbody = data.map((row, index) => {
        const evPercent = row.evPercent || 0;
        const kellyQuarter = row.kellyQuarter || 0;
        
        return `
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
            <td style="font-weight: 600; color: ${row.odds1 > 0 ? '#f87171' : '#34d399'}; font-size: 0.9rem;">
                ${formatOdds(row.odds1 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.9rem;">
                ${((row.modelProb1 || 0) * 100).toFixed(1)}%
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
            <td style="font-weight: 600; color: ${row.odds2 > 0 ? '#f87171' : '#34d399'}; font-size: 0.9rem;">
                ${formatOdds(row.odds2 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.9rem;">
                ${((row.modelProb2 || 0) * 100).toFixed(1)}%
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.9rem;">
                ${((row.parlayProb || 0) * 100).toFixed(1)}%
            </td>
            <td style="font-weight: 600; color: ${row.parlayOdds > 0 ? '#f87171' : '#34d399'}; font-size: 0.9rem;">
                ${formatOdds(row.parlayOdds || -137)}
            </td>
            <td class="ev-cell ${getEVClass(evPercent)}">${evPercent.toFixed(2)}%</td>
            <td class="kelly-cell">${(kellyQuarter * 100).toFixed(1)}%</td>
        </tr>
    `}).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

function renderTriosTable(data) {
    const thead = `
        <tr>
            <th style="width: 2%">#</th>
            <th style="width: 9%">Player 1</th>
            <th style="width: 4%">Line</th>
            <th style="width: 4%">Proj.</th>
            <th style="width: 4%">Odds</th>
            <th style="width: 5%">Model Prob</th>
            <th style="width: 9%">Player 2</th>
            <th style="width: 4%">Line</th>
            <th style="width: 4%">Proj.</th>
            <th style="width: 4%">Odds</th>
            <th style="width: 5%">Model Prob</th>
            <th style="width: 9%">Player 3</th>
            <th style="width: 4%">Line</th>
            <th style="width: 4%">Proj.</th>
            <th style="width: 4%">Odds</th>
            <th style="width: 5%">Model Prob</th>
            <th style="width: 5%">Parlay Prob</th>
            <th style="width: 5%">Parlay Odds</th>
            <th style="width: 5%">EV%</th>
            <th style="width: 5%">Kelly</th>
        </tr>
    `;

    const tbody = data.map((row, index) => {
        const evPercent = row.evPercent || 0;
        const kellyQuarter = row.kellyQuarter || 0;
        
        return `
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
            <td style="font-weight: 600; color: ${row.odds1 > 0 ? '#f87171' : '#34d399'}; font-size: 0.85rem;">
                ${formatOdds(row.odds1 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.modelProb1 || 0) * 100).toFixed(1)}%
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
            <td style="font-weight: 600; color: ${row.odds2 > 0 ? '#f87171' : '#34d399'}; font-size: 0.85rem;">
                ${formatOdds(row.odds2 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.modelProb2 || 0) * 100).toFixed(1)}%
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
            <td style="font-weight: 600; color: ${row.odds3 > 0 ? '#f87171' : '#34d399'}; font-size: 0.85rem;">
                ${formatOdds(row.odds3 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.modelProb3 || 0) * 100).toFixed(1)}%
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.parlayProb || 0) * 100).toFixed(1)}%
            </td>
            <td style="font-weight: 600; color: ${row.parlayOdds > 0 ? '#f87171' : '#34d399'}; font-size: 0.85rem;">
                ${formatOdds(row.parlayOdds || -137)}
            </td>
            <td class="ev-cell ${getEVClass(evPercent)}">${evPercent.toFixed(2)}%</td>
            <td class="kelly-cell">${(kellyQuarter * 100).toFixed(1)}%</td>
        </tr>
    `}).join('');

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
                <div class="stat-label">Odds</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">American odds format. Positive odds (red) show potential profit on $100 bet. Negative odds (green) show amount needed to bet to win $100.</div>
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

