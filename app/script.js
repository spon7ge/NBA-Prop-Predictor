const prizepicksSinglesData = [
    {name: "Luka Doncic", bookmaker: "Bovada", line: 28.5, prediction: 20.35, side: "Under", odds: 205, recommendation: 1, ev: 15.58, roi: 155.8, kelly: 0.76, sigma: "High"},
    {name: "Luka Doncic", bookmaker: "Bovada", line: 29.5, prediction: 20.35, side: "Under", odds: 170, recommendation: 1, ev: 13.47, roi: 134.7, kelly: 0.793, sigma: "High"},
    {name: "Luka Doncic", bookmaker: "Bovada", line: 30.5, prediction: 20.35, side: "Under", odds: 145, recommendation: 1, ev: 11.77, roi: 117.7, kelly: 0.812, sigma: "High"},
    {name: "Luka Doncic", bookmaker: "Bovada", line: 31.5, prediction: 20.35, side: "Under", odds: 120, recommendation: 1, ev: 10.02, roi: 100.2, kelly: 0.835, sigma: "High"},
    {name: "Luka Doncic", bookmaker: "BetRivers", line: 32.5, prediction: 20.35, side: "Under", odds: 114, recommendation: 1, ev: 9.94, roi: 99.4, kelly: 0.872, sigma: "High"},
    {name: "Luka Doncic", bookmaker: "Bovada", line: 32.5, prediction: 20.35, side: "Under", odds: 100, recommendation: 1, ev: 8.58, roi: 85.8, kelly: 0.858, sigma: "High"},
    {name: "Luka Doncic", bookmaker: "BetRivers", line: 33.5, prediction: 20.35, side: "Under", odds: -104, recommendation: 1, ev: 8.53, roi: 85.3, kelly: 0.887, sigma: "High"},
    {name: "Luka Doncic", bookmaker: "Bovada", line: 33.5, prediction: 20.35, side: "Under", odds: -110, recommendation: 1, ev: 7.99, roi: 79.9, kelly: 0.879, sigma: "High"},
    {name: "Luka Doncic", bookmaker: "DraftKings", line: 34.5, prediction: 20.35, side: "Under", odds: -122, recommendation: 1, ev: 7.4, roi: 74.0, kelly: 0.903, sigma: "High"},
    {name: "Simone Fontecchio", bookmaker: "Bovada", line: 10.5, prediction: 12.63, side: "Over", odds: 170, recommendation: 1, ev: 7.39, roi: 73.9, kelly: 0.435, sigma: "High"}
];

const prizepicksPairsData = [
    {"name1": "T.J. McConnell", "name2": "Luka Don\u010di\u0107", "line1": 10.0, "line2": 32.5, "side1": "under", "side2": "under", "recommendation": 1, "ev": 8.88, "kelly": 0.444, "sigma1": "Low", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Jock Landale", "name2": "Luka Don\u010di\u0107", "line1": 7.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 8.58, "kelly": 0.429, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Scottie Barnes", "name2": "Luka Don\u010di\u0107", "line1": 18.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 7.28, "kelly": 0.364, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Tony Bradley", "name2": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 7.01, "kelly": 0.35, "sigma1": "Med", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Jock Landale", "name2": "T.J. McConnell", "line1": 7.5, "line2": 10.0, "side1": "over", "side2": "under", "recommendation": 1, "ev": 6.81, "kelly": 0.341, "sigma1": "High", "sigma2": "Low", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Spencer Jones", "name2": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 32.5, "side1": "under", "side2": "under", "recommendation": 1, "ev": 6.76, "kelly": 0.338, "sigma1": "Low", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Donovan Mitchell", "name2": "Luka Don\u010di\u0107", "line1": 28.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 6.57, "kelly": 0.328, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Julius Randle", "name2": "Luka Don\u010di\u0107", "line1": 21.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 5.97, "kelly": 0.298, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Jaden McDaniels", "name2": "Luka Don\u010di\u0107", "line1": 13.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 5.88, "kelly": 0.294, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Cam Spencer", "name2": "Luka Don\u010di\u0107", "line1": 7.0, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 5.85, "kelly": 0.293, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
];const prizepicksTriosData = [
    {"name1": "Donovan Mitchell", "name2": "Scottie Barnes", "name3": "Luka Don\u010di\u0107", "line1": 28.5, "line2": 18.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 12.01, "kelly": 0.24, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Cam Spencer", "name2": "Scottie Barnes", "name3": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 18.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 11.96, "kelly": 0.239, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Scottie Barnes", "name2": "Spencer Jones", "name3": "Luka Don\u010di\u0107", "line1": 18.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 11.93, "kelly": 0.239, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Cam Spencer", "name2": "Spencer Jones", "name3": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 11.91, "kelly": 0.238, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Scottie Barnes", "name2": "Tony Bradley", "name3": "Luka Don\u010di\u0107", "line1": 18.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 11.87, "kelly": 0.237, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Cam Spencer", "name2": "Tony Bradley", "name3": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 11.81, "kelly": 0.236, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Donovan Mitchell", "name2": "Tony Bradley", "name3": "Luka Don\u010di\u0107", "line1": 28.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 11.79, "kelly": 0.236, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Spencer Jones", "name2": "Luka Don\u010di\u0107", "name3": "Marcus Smart", "line1": 6.5, "line2": 32.5, "line3": 9.5, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 11.44, "kelly": 0.229, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Tony Bradley", "name2": "Spencer Jones", "name3": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 11.27, "kelly": 0.225, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Tony Bradley", "name2": "Luka Don\u010di\u0107", "name3": "Marcus Smart", "line1": 6.5, "line2": 32.5, "line3": 9.5, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 11.25, "kelly": 0.225, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Spencer Jones", "name2": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 32.5, "side1": "under", "side2": "under", "recommendation": 1, "ev": 6.99, "kelly": 0.349, "sigma1": "Low", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Scottie Barnes", "name2": "Luka Don\u010di\u0107", "line1": 18.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 6.98, "kelly": 0.349, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Tony Bradley", "name2": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 6.93, "kelly": 0.347, "sigma1": "Med", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Donovan Mitchell", "name2": "Luka Don\u010di\u0107", "line1": 28.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 6.86, "kelly": 0.343, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Cam Spencer", "name2": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 6.71, "kelly": 0.335, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Julius Randle", "name2": "Luka Don\u010di\u0107", "line1": 21.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 5.98, "kelly": 0.299, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Jaylen Clark", "name2": "Luka Don\u010di\u0107", "line1": 4.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 5.88, "kelly": 0.294, "sigma1": "Low", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Ajay Mitchell", "name2": "Luka Don\u010di\u0107", "line1": 14.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 5.77, "kelly": 0.289, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Nikola Joki\u0107", "name2": "Luka Don\u010di\u0107", "line1": 27.5, "line2": 32.5, "side1": "under", "side2": "under", "recommendation": 1, "ev": 5.75, "kelly": 0.287, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Sam Merrill", "name2": "Luka Don\u010di\u0107", "line1": 11.5, "line2": 32.5, "side1": "over", "side2": "under", "recommendation": 1, "ev": 5.44, "kelly": 0.272, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
];const underdogTriosData = [
    {"name1": "Donovan Mitchell", "name2": "Scottie Barnes", "name3": "Luka Don\u010di\u0107", "line1": 28.5, "line2": 18.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 12.01, "kelly": 0.24, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Cam Spencer", "name2": "Scottie Barnes", "name3": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 18.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 11.96, "kelly": 0.239, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Scottie Barnes", "name2": "Spencer Jones", "name3": "Luka Don\u010di\u0107", "line1": 18.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 11.93, "kelly": 0.239, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Cam Spencer", "name2": "Spencer Jones", "name3": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 11.91, "kelly": 0.238, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Scottie Barnes", "name2": "Tony Bradley", "name3": "Luka Don\u010di\u0107", "line1": 18.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 11.87, "kelly": 0.237, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Cam Spencer", "name2": "Tony Bradley", "name3": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 11.81, "kelly": 0.236, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Donovan Mitchell", "name2": "Tony Bradley", "name3": "Luka Don\u010di\u0107", "line1": 28.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 11.79, "kelly": 0.236, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Spencer Jones", "name2": "Luka Don\u010di\u0107", "name3": "Marcus Smart", "line1": 6.5, "line2": 32.5, "line3": 9.5, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 11.44, "kelly": 0.229, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Tony Bradley", "name2": "Spencer Jones", "name3": "Luka Don\u010di\u0107", "line1": 6.5, "line2": 6.5, "line3": 32.5, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 11.27, "kelly": 0.225, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Tony Bradley", "name2": "Luka Don\u010di\u0107", "name3": "Marcus Smart", "line1": 6.5, "line2": 32.5, "line3": 9.5, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 11.25, "kelly": 0.225, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
];// This is a large data file - I'll create a simplified version that includes all the hit rates data
// For brevity, I'll include a condensed version with the key structures
const prizepicksPointsHitRates = [
    {"name": "Harrison Barnes", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.847, "underPct": 0.153},
    {"name": "Jaden McDaniels", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.845, "underPct": 0.155},
    {"name": "Tari Eason", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.805, "underPct": 0.195},
    {"name": "Tre Mann", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.779, "underPct": 0.221},
    {"name": "Rui Hachimura", "line": 12.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.765, "underPct": 0.235},
    {"name": "Austin Reaves", "line": 24.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.758, "underPct": 0.242},
    {"name": "Norman Powell", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.753, "underPct": 0.247},
    {"name": "Moses Moody", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.749, "underPct": 0.251},
    {"name": "Jalen Duren", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.746, "underPct": 0.254},
    {"name": "Deni Avdija", "line": 23.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.736, "underPct": 0.264},
    {"name": "Julian Champagnie", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.721, "underPct": 0.279},
    {"name": "Reed Sheppard", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.71, "underPct": 0.29},
    {"name": "Ryan Rollins", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.709, "underPct": 0.291},
    {"name": "Pelle Larsson", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.707, "underPct": 0.293},
    {"name": "Luke Kornet", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.7, "underPct": 0.3},
    {"name": "Sion James", "line": 6.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.698, "underPct": 0.302},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.695, "underPct": 0.305},
    {"name": "Miles Bridges", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.691, "underPct": 0.309},
    {"name": "Shaedon Sharpe", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.677, "underPct": 0.323},
    {"name": "Max Christie", "line": 11.5, "l5": 1.0, "l10": 0.9, "l15": 0.6, "overPct": 0.673, "underPct": 0.327},
    {"name": "Stephon Castle", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.667, "underPct": 0.333},
    {"name": "Cade Cunningham", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.664, "underPct": 0.336},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.663, "underPct": 0.337},
    {"name": "LaMelo Ball", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.653, "underPct": 0.347},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.638, "underPct": 0.362},
    {"name": "Marcus Smart", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.631, "underPct": 0.369},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.629, "underPct": 0.371},
    {"name": "Quentin Grimes", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.612, "underPct": 0.388},
    {"name": "Kon Knueppel", "line": 15.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.611, "underPct": 0.389},
    {"name": "Jaylen Clark", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.607, "underPct": 0.393},
    {"name": "Russell Westbrook", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.605, "underPct": 0.395},
    {"name": "Domantas Sabonis", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.588, "underPct": 0.412},
    {"name": "Trendon Watford", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.588, "underPct": 0.412},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.587, "underPct": 0.413},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.584, "underPct": 0.416},
    {"name": "Josh Okogie", "line": 7.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.582, "underPct": 0.418},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.573, "underPct": 0.427},
    {"name": "Daniel Gafford", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.573, "underPct": 0.427},
    {"name": "Kris Dunn", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.567, "underPct": 0.433},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.562, "underPct": 0.438},
    {"name": "Myles Turner", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.558, "underPct": 0.442},
    {"name": "Brandin Podziemski", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.557, "underPct": 0.443},
    {"name": "Andrew Wiggins", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.552, "underPct": 0.448},
    {"name": "Deandre Ayton", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.543, "underPct": 0.457},
    {"name": "Jarred Vanderbilt", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.524, "underPct": 0.476},
    {"name": "Jalen Wilson", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.517, "underPct": 0.483},
    {"name": "Tyrese Maxey", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.513, "underPct": 0.487},
    {"name": "Buddy Hield", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.507, "underPct": 0.493},
    {"name": "Jonathan Kuminga", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.506, "underPct": 0.494},
    {"name": "De'Aaron Fox", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.486, "underPct": 0.514},
    {"name": "Brook Lopez", "line": 7.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.482, "underPct": 0.518},
    {"name": "DeMar DeRozan", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.482, "underPct": 0.518},
    {"name": "Jrue Holiday", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.464, "underPct": 0.536},
    {"name": "Stephen Curry", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.463, "underPct": 0.537},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.461, "underPct": 0.539},
    {"name": "Bobby Portis", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.456, "underPct": 0.544},
    {"name": "Giannis Antetokounmpo", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.45, "underPct": 0.55},
    {"name": "Davion Mitchell", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.447, "underPct": 0.553},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.445, "underPct": 0.555},
    {"name": "James Harden", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.425, "underPct": 0.575},
    {"name": "Alperen Sengun", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.42, "underPct": 0.58},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.42, "underPct": 0.58},
    {"name": "Saddiq Bey", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.414, "underPct": 0.586},
    {"name": "P.J. Washington", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.41, "underPct": 0.59},
    {"name": "Anthony Davis", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.408, "underPct": 0.592},
    {"name": "Devin Vassell", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.404, "underPct": 0.596},
    {"name": "Collin Sexton", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.401, "underPct": 0.599},
    {"name": "Derik Queen", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.396, "underPct": 0.604},
    {"name": "Landry Shamet", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.394, "underPct": 0.606},
    {"name": "Jerami Grant", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.39, "underPct": 0.61},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.388, "underPct": 0.612},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.387, "underPct": 0.613},
    {"name": "Tristan da Silva", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.383, "underPct": 0.617},
    {"name": "Anthony Black", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.383, "underPct": 0.617},
    {"name": "Terance Mann", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.377, "underPct": 0.623},
    {"name": "Kevin Durant", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.37, "underPct": 0.63},
    {"name": "Kris Murray", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.365, "underPct": 0.635},
    {"name": "Ivica Zubac", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.364, "underPct": 0.636},
    {"name": "Jeremiah Fears", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.358, "underPct": 0.642},
    {"name": "Tyrese Martin", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.348, "underPct": 0.652},
    {"name": "Victor Wembanyama", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.342, "underPct": 0.658},
    {"name": "Draymond Green", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.335, "underPct": 0.665},
    {"name": "Noah Clowney", "line": 12.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.331, "underPct": 0.669},
    {"name": "Andre Drummond", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.329, "underPct": 0.671},
    {"name": "Josh Hart", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.318, "underPct": 0.682},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.316, "underPct": 0.684},
    {"name": "Cole Anthony", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.31, "underPct": 0.69},
    {"name": "Naz Reid", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.302, "underPct": 0.698},
    {"name": "OG Anunoby", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.301, "underPct": 0.699},
    {"name": "Mitchell Robinson", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.296, "underPct": 0.704},
    {"name": "VJ Edgecombe", "line": 13.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.295, "underPct": 0.705},
    {"name": "Al Horford", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.285, "underPct": 0.715},
    {"name": "Franz Wagner", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.276, "underPct": 0.724},
    {"name": "Ziaire Williams", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.275, "underPct": 0.725},
    {"name": "John Collins", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.27, "underPct": 0.73},
    {"name": "Mikal Bridges", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.259, "underPct": 0.741},
    {"name": "Nicolas Batum", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.227, "underPct": 0.773},
    {"name": "Jalen Suggs", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.227, "underPct": 0.773},
    {"name": "Karl-Anthony Towns", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luka Doncic", "line": 33.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.169, "underPct": 0.831},
    {"name": "Jonathan Isaac", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.135, "underPct": 0.865},
    {"name": "Desmond Bane", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.079, "underPct": 0.921},
    {"name": "Miles McBride", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.051, "underPct": 0.949},
];const prizepicksAssistsHitRates = [
    {"name": "Stephon Castle", "line": 7.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.754, "underPct": 0.246},
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.664, "underPct": 0.336},
    {"name": "Austin Reaves", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.65, "underPct": 0.35},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.641, "underPct": 0.359},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.62, "underPct": 0.38},
    {"name": "Kon Knueppel", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.619, "underPct": 0.381},
    {"name": "Cole Anthony", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.605, "underPct": 0.395},
    {"name": "Dru Smith", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Alperen Sengun", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.585, "underPct": 0.415},
    {"name": "Tyrese Maxey", "line": 7.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.567, "underPct": 0.433},
    {"name": "Cade Cunningham", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.56, "underPct": 0.44},
    {"name": "Ryan Rollins", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.56, "underPct": 0.44},
    {"name": "Amen Thompson", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.526, "underPct": 0.474},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.521, "underPct": 0.479},
    {"name": "Deni Avdija", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.52, "underPct": 0.48},
    {"name": "Victor Wembanyama", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.501, "underPct": 0.499},
    {"name": "Marcus Smart", "line": 3.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.501, "underPct": 0.499},
    {"name": "Bobby Portis", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.479, "underPct": 0.521},
    {"name": "Devin Vassell", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.469, "underPct": 0.531},
    {"name": "Al Horford", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.469, "underPct": 0.531},
    {"name": "Jalen Suggs", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.444, "underPct": 0.556},
    {"name": "Davion Mitchell", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.426, "underPct": 0.574},
    {"name": "Kevon Looney", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.418, "underPct": 0.582},
    {"name": "Mikal Bridges", "line": 5.0, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.401, "underPct": 0.599},
    {"name": "Jeremiah Fears", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.391, "underPct": 0.609},
    {"name": "Domantas Sabonis", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.375, "underPct": 0.625},
    {"name": "Brook Lopez", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.358, "underPct": 0.642},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.355, "underPct": 0.645},
    {"name": "Trey Murphy III", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.355, "underPct": 0.645},
    {"name": "De'Aaron Fox", "line": 5.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.351, "underPct": 0.649},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.348, "underPct": 0.652},
    {"name": "Stephen Curry", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.307, "underPct": 0.693},
    {"name": "James Harden", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.302, "underPct": 0.698},
    {"name": "Cooper Flagg", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.287, "underPct": 0.713},
    {"name": "Draymond Green", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.267, "underPct": 0.733},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.248, "underPct": 0.752},
    {"name": "Giannis Antetokounmpo", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.248, "underPct": 0.752},
    {"name": "Franz Wagner", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.24, "underPct": 0.76},
    {"name": "Goga Bitadze", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.183, "underPct": 0.817},
    {"name": "Josh Hart", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.175, "underPct": 0.825},
    {"name": "Luka Doncic", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.116, "underPct": 0.884},
];const prizepicksReboundsHitRates = [
    {"name": "LaMelo Ball", "line": 5.0, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.808, "underPct": 0.192},
    {"name": "Kon Knueppel", "line": 5.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.777, "underPct": 0.223},
    {"name": "Jonathan Kuminga", "line": 4.0, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.673, "underPct": 0.327},
    {"name": "Alperen Sengun", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.657, "underPct": 0.343},
    {"name": "Julian Champagnie", "line": 3.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.643, "underPct": 0.357},
    {"name": "Max Christie", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.635, "underPct": 0.365},
    {"name": "VJ Edgecombe", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.63, "underPct": 0.37},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.623, "underPct": 0.377},
    {"name": "Stephon Castle", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.622, "underPct": 0.378},
    {"name": "Mitchell Robinson", "line": 7.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.621, "underPct": 0.379},
    {"name": "Miles Bridges", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.615, "underPct": 0.385},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.614, "underPct": 0.386},
    {"name": "Trendon Watford", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.608, "underPct": 0.392},
    {"name": "P.J. Washington", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.594, "underPct": 0.406},
    {"name": "Shaedon Sharpe", "line": 4.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.581, "underPct": 0.419},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.578, "underPct": 0.422},
    {"name": "Franz Wagner", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.576, "underPct": 0.424},
    {"name": "Day'Ron Sharpe", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.575, "underPct": 0.425},
    {"name": "Cooper Flagg", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.571, "underPct": 0.429},
    {"name": "Karl-Anthony Towns", "line": 12.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.567, "underPct": 0.433},
    {"name": "Victor Wembanyama", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.567, "underPct": 0.433},
    {"name": "Anthony Davis", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.563, "underPct": 0.437},
    {"name": "Jrue Holiday", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.563, "underPct": 0.437},
    {"name": "Amen Thompson", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.557, "underPct": 0.443},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.553, "underPct": 0.447},
    {"name": "Ivica Zubac", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.548, "underPct": 0.452},
    {"name": "Austin Reaves", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.526, "underPct": 0.474},
    {"name": "James Harden", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.523, "underPct": 0.477},
    {"name": "Saddiq Bey", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.515, "underPct": 0.485},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.509, "underPct": 0.491},
    {"name": "Devin Vassell", "line": 3.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.508, "underPct": 0.492},
    {"name": "Keldon Johnson", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.493, "underPct": 0.507},
    {"name": "Jalen Duren", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.492, "underPct": 0.508},
    {"name": "OG Anunoby", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.487, "underPct": 0.513},
    {"name": "Daniel Gafford", "line": 5.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.479, "underPct": 0.521},
    {"name": "Marcus Smart", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.472, "underPct": 0.528},
    {"name": "Julius Randle", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.469, "underPct": 0.531},
    {"name": "Donovan Clingan", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.467, "underPct": 0.533},
    {"name": "Deni Avdija", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.457, "underPct": 0.543},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.446, "underPct": 0.554},
    {"name": "Brandon Williams", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.441, "underPct": 0.559},
    {"name": "Jaden McDaniels", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.436, "underPct": 0.564},
    {"name": "Tari Eason", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.434, "underPct": 0.566},
    {"name": "Dereck Lively II", "line": 6.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.431, "underPct": 0.569},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.43, "underPct": 0.57},
    {"name": "Cade Cunningham", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.416, "underPct": 0.584},
    {"name": "Kevin Durant", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.409, "underPct": 0.591},
    {"name": "Kel'el Ware", "line": 10.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.401, "underPct": 0.599},
    {"name": "Al Horford", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.379, "underPct": 0.621},
    {"name": "Draymond Green", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.367, "underPct": 0.633},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.363, "underPct": 0.637},
    {"name": "Anthony Black", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.355, "underPct": 0.645},
    {"name": "Myles Turner", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.351, "underPct": 0.649},
    {"name": "Domantas Sabonis", "line": 13.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.343, "underPct": 0.657},
    {"name": "Derik Queen", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.339, "underPct": 0.661},
    {"name": "Giannis Antetokounmpo", "line": 11.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.321, "underPct": 0.679},
    {"name": "Andrew Wiggins", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.319, "underPct": 0.681},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.308, "underPct": 0.692},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.298, "underPct": 0.702},
    {"name": "Rui Hachimura", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.297, "underPct": 0.703},
    {"name": "Bobby Portis", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.292, "underPct": 0.708},
    {"name": "John Collins", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.29, "underPct": 0.71},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.289, "underPct": 0.711},
    {"name": "Naz Reid", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.285, "underPct": 0.715},
    {"name": "Tristan da Silva", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.279, "underPct": 0.721},
    {"name": "Deandre Ayton", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.257, "underPct": 0.743},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.25, "underPct": 0.75},
    {"name": "Josh Hart", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.229, "underPct": 0.771},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.229, "underPct": 0.771},
    {"name": "Brook Lopez", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.226, "underPct": 0.774},
    {"name": "Stephen Curry", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.217, "underPct": 0.783},
    {"name": "Andre Drummond", "line": 12.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.152, "underPct": 0.848},
    {"name": "Goga Bitadze", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.148, "underPct": 0.852},
    {"name": "Luka Doncic", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.129, "underPct": 0.871},
    {"name": "Jonathan Isaac", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.126, "underPct": 0.874},
];const prizepicksBlocksHitRates = [
    {"name": "Jalen Duren", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.316, "underPct": 0.684},
    {"name": "Alperen Sengun", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.496, "underPct": 0.504},
    {"name": "Deni Avdija", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.561, "underPct": 0.439},
    {"name": "Steven Adams", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.479, "underPct": 0.521},
    {"name": "Tari Eason", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.387, "underPct": 0.613},
    {"name": "Derik Queen", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.396, "underPct": 0.604},
    {"name": "Rudy Gobert", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.578, "underPct": 0.422},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.336, "underPct": 0.664},
    {"name": "Al Horford", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.64, "underPct": 0.36},
];const prizepicksStealsHitRates = [
    {"name": "Day'Ron Sharpe", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.601, "underPct": 0.399},
    {"name": "Ziaire Williams", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jordan Clarkson", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.32, "underPct": 0.68},
    {"name": "Trendon Watford", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.319, "underPct": 0.681},
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.506, "underPct": 0.494},
    {"name": "Bobby Portis", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.321, "underPct": 0.679},
    {"name": "Kyle Kuzma", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.357, "underPct": 0.643},
    {"name": "Miles Bridges", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.457, "underPct": 0.543},
    {"name": "Tre Mann", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.493, "underPct": 0.507},
    {"name": "Kon Knueppel", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.437, "underPct": 0.563},
    {"name": "Saddiq Bey", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.458, "underPct": 0.542},
    {"name": "Zach LaVine", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.401, "underPct": 0.599},
    {"name": "Cooper Flagg", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.445, "underPct": 0.555},
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.494, "underPct": 0.506},
    {"name": "Dereck Lively II", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.503, "underPct": 0.497},
];const underdogPointsHitRates = [
    {"name": "Jake LaRavia", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.865, "underPct": 0.135},
    {"name": "Jaden McDaniels", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.845, "underPct": 0.155},
    {"name": "Julian Champagnie", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.828, "underPct": 0.172},
    {"name": "Tari Eason", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.805, "underPct": 0.195},
    {"name": "Tre Mann", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.779, "underPct": 0.221},
    {"name": "Rui Hachimura", "line": 12.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.765, "underPct": 0.235},
    {"name": "Austin Reaves", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.758, "underPct": 0.242},
    {"name": "Shaedon Sharpe", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.751, "underPct": 0.249},
    {"name": "Steven Adams", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.751, "underPct": 0.249},
    {"name": "Jalen Duren", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.746, "underPct": 0.254},
    {"name": "LaMelo Ball", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.728, "underPct": 0.272},
    {"name": "Mike Conley", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.718, "underPct": 0.282},
    {"name": "Reed Sheppard", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.71, "underPct": 0.29},
    {"name": "Ryan Rollins", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.709, "underPct": 0.291},
    {"name": "Pelle Larsson", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.707, "underPct": 0.293},
    {"name": "Luke Kornet", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.7, "underPct": 0.3},
    {"name": "Sion James", "line": 6.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.698, "underPct": 0.302},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.695, "underPct": 0.305},
    {"name": "Miles Bridges", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.691, "underPct": 0.309},
    {"name": "Norman Powell", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.685, "underPct": 0.315},
    {"name": "Max Christie", "line": 11.5, "l5": 1.0, "l10": 0.9, "l15": 0.6, "overPct": 0.673, "underPct": 0.327},
    {"name": "Keldon Johnson", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.664, "underPct": 0.336},
    {"name": "Cade Cunningham", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.664, "underPct": 0.336},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.663, "underPct": 0.337},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.652, "underPct": 0.348},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.638, "underPct": 0.362},
    {"name": "Marcus Smart", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.631, "underPct": 0.369},
    {"name": "Quentin Grimes", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.612, "underPct": 0.388},
    {"name": "Kon Knueppel", "line": 15.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.611, "underPct": 0.389},
    {"name": "Jaylen Clark", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.607, "underPct": 0.393},
    {"name": "Russell Westbrook", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.605, "underPct": 0.395},
    {"name": "Domantas Sabonis", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.588, "underPct": 0.412},
    {"name": "Trendon Watford", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.588, "underPct": 0.412},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.584, "underPct": 0.416},
    {"name": "Daniel Gafford", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.573, "underPct": 0.427},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.573, "underPct": 0.427},
    {"name": "Kris Dunn", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.567, "underPct": 0.433},
    {"name": "Toumani Camara", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.562, "underPct": 0.438},
    {"name": "Myles Turner", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.558, "underPct": 0.442},
    {"name": "Brandin Podziemski", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.557, "underPct": 0.443},
    {"name": "Andrew Wiggins", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.552, "underPct": 0.448},
    {"name": "Deandre Ayton", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.543, "underPct": 0.457},
    {"name": "Amen Thompson", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.539, "underPct": 0.461},
    {"name": "Jalen Wilson", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.517, "underPct": 0.483},
    {"name": "Anthony Davis", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.495, "underPct": 0.505},
    {"name": "Malik Monk", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.49, "underPct": 0.51},
    {"name": "DeMar DeRozan", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.482, "underPct": 0.518},
    {"name": "Brook Lopez", "line": 7.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.482, "underPct": 0.518},
    {"name": "Bobby Portis", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.456, "underPct": 0.544},
    {"name": "Giannis Antetokounmpo", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.45, "underPct": 0.55},
    {"name": "Goga Bitadze", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.444, "underPct": 0.556},
    {"name": "Noah Clowney", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.442, "underPct": 0.558},
    {"name": "Josh Hart", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.433, "underPct": 0.567},
    {"name": "James Harden", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.425, "underPct": 0.575},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.422, "underPct": 0.578},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.42, "underPct": 0.58},
    {"name": "Alperen Sengun", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.42, "underPct": 0.58},
    {"name": "Saddiq Bey", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.414, "underPct": 0.586},
    {"name": "Collin Sexton", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.401, "underPct": 0.599},
    {"name": "Landry Shamet", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.394, "underPct": 0.606},
    {"name": "Jerami Grant", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.39, "underPct": 0.61},
    {"name": "Stephen Curry", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.386, "underPct": 0.614},
    {"name": "Terance Mann", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.377, "underPct": 0.623},
    {"name": "Tyrese Maxey", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.371, "underPct": 0.629},
    {"name": "Kevin Durant", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.37, "underPct": 0.63},
    {"name": "Jeremiah Fears", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.358, "underPct": 0.642},
    {"name": "Tyrese Martin", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.348, "underPct": 0.652},
    {"name": "Franz Wagner", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.347, "underPct": 0.653},
    {"name": "Kevon Looney", "line": 3.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.341, "underPct": 0.659},
    {"name": "Anthony Edwards", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.318, "underPct": 0.682},
    {"name": "Cole Anthony", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.31, "underPct": 0.69},
    {"name": "Naz Reid", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.302, "underPct": 0.698},
    {"name": "OG Anunoby", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.301, "underPct": 0.699},
    {"name": "Mitchell Robinson", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.3, "underPct": 0.7},
    {"name": "VJ Edgecombe", "line": 13.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.295, "underPct": 0.705},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.293, "underPct": 0.707},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.276, "underPct": 0.724},
    {"name": "John Collins", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.27, "underPct": 0.73},
    {"name": "Nicolas Batum", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.227, "underPct": 0.773},
    {"name": "Jalen Suggs", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.227, "underPct": 0.773},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.215, "underPct": 0.785},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.209, "underPct": 0.791},
    {"name": "Karl-Anthony Towns", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luka Doncic", "line": 33.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.169, "underPct": 0.831},
    {"name": "Jonathan Isaac", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.135, "underPct": 0.865},
    {"name": "Desmond Bane", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.079, "underPct": 0.921},
    {"name": "Miles McBride", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.051, "underPct": 0.949},
];const underdogAssistsHitRates = [
    {"name": "Cade Cunningham", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.678, "underPct": 0.322},
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.664, "underPct": 0.336},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.62, "underPct": 0.38},
    {"name": "Dru Smith", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Victor Wembanyama", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.501, "underPct": 0.499},
    {"name": "Bobby Portis", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.479, "underPct": 0.521},
    {"name": "Devin Vassell", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.469, "underPct": 0.531},
    {"name": "Al Horford", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.469, "underPct": 0.531},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.449, "underPct": 0.551},
    {"name": "OG Anunoby", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.434, "underPct": 0.566},
    {"name": "Jeremiah Fears", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.391, "underPct": 0.609},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.348, "underPct": 0.652},
    {"name": "Jalen Duren", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.303, "underPct": 0.697},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.248, "underPct": 0.752},
    {"name": "Brandon Williams", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.236, "underPct": 0.764},
];const underdogReboundsHitRates = [
    {"name": "Kon Knueppel", "line": 5.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.777, "underPct": 0.223},
    {"name": "Alperen Sengun", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.657, "underPct": 0.343},
    {"name": "Julian Champagnie", "line": 3.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.643, "underPct": 0.357},
    {"name": "Max Christie", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.635, "underPct": 0.365},
    {"name": "Jake LaRavia", "line": 3.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.633, "underPct": 0.367},
    {"name": "Mitchell Robinson", "line": 7.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.621, "underPct": 0.379},
    {"name": "Miles Bridges", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.615, "underPct": 0.385},
    {"name": "Trendon Watford", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.608, "underPct": 0.392},
    {"name": "P.J. Washington", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.594, "underPct": 0.406},
    {"name": "Sion James", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.583, "underPct": 0.417},
    {"name": "Cooper Flagg", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.571, "underPct": 0.429},
    {"name": "Victor Wembanyama", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.567, "underPct": 0.433},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.553, "underPct": 0.447},
    {"name": "James Harden", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.523, "underPct": 0.477},
    {"name": "Saddiq Bey", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.515, "underPct": 0.485},
    {"name": "Myles Turner", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.51, "underPct": 0.49},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.509, "underPct": 0.491},
    {"name": "Marcus Smart", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.472, "underPct": 0.528},
    {"name": "Deni Avdija", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.457, "underPct": 0.543},
    {"name": "Kevon Looney", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.452, "underPct": 0.548},
    {"name": "Domantas Sabonis", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.45, "underPct": 0.55},
    {"name": "Cade Cunningham", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.416, "underPct": 0.584},
    {"name": "Jalen Wilson", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.358, "underPct": 0.642},
    {"name": "Anthony Black", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.355, "underPct": 0.645},
    {"name": "Daniel Gafford", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.321, "underPct": 0.679},
    {"name": "Rui Hachimura", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.297, "underPct": 0.703},
    {"name": "Deandre Ayton", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.257, "underPct": 0.743},
    {"name": "Brook Lopez", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.226, "underPct": 0.774},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.152, "underPct": 0.848},
    {"name": "Luka Doncic", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.129, "underPct": 0.871},
];const underdogBlocksHitRates = [
    {"name": "Jalen Duren", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.316, "underPct": 0.684},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.452, "underPct": 0.548},
    {"name": "Rudy Gobert", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.578, "underPct": 0.422},
];const underdogStealsHitRates = [
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.506, "underPct": 0.494},
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
            <th style="width: 4%">#</th>
            <th style="width: 20%">Player 1</th>
            <th style="width: 8%">Line 1</th>
            <th style="width: 20%">Player 2</th>
            <th style="width: 8%">Line 2</th>
            <th style="width: 10%">EV %</th>
            <th style="width: 10%">Kelly</th>
            <th style="width: 15%">Sigma</th>
            <th style="width: 5%">Rec</th>
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
                        <span class="hit-rate ${getHitRateClass(row.hitRate1)} tooltip" data-tooltip="L-5: ${(row.l5_1*100).toFixed(0)}% | L-15: ${(row.l15_1*100).toFixed(0)}%">${row.hitRate1.toFixed(1)}%</span>
                        ${getTrendArrow(row.l5_1, row.l15_1)}
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line1}</td>
            <td>
                <div class="player-cell">
                    <span class="player-name">${row.name2}</span>
                    <div class="player-stats">
                        <span class="side-badge side-${row.side2}">${row.side2}</span>
                        <span class="hit-rate ${getHitRateClass(row.hitRate2)} tooltip" data-tooltip="L-5: ${(row.l5_2*100).toFixed(0)}% | L-15: ${(row.l15_2*100).toFixed(0)}%">${row.hitRate2.toFixed(1)}%</span>
                        ${getTrendArrow(row.l5_2, row.l15_2)}
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line2}</td>
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
            <th style="width: 3%">#</th>
            <th style="width: 18%">Player 1</th>
            <th style="width: 6%">Line 1</th>
            <th style="width: 18%">Player 2</th>
            <th style="width: 6%">Line 2</th>
            <th style="width: 18%">Player 3</th>
            <th style="width: 6%">Line 3</th>
            <th style="width: 8%">EV %</th>
            <th style="width: 8%">Kelly</th>
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
                        <span class="hit-rate ${getHitRateClass(row.hitRate1)} tooltip" data-tooltip="L-5: ${(row.l5_1*100).toFixed(0)}% | L-15: ${(row.l15_1*100).toFixed(0)}%">${row.hitRate1.toFixed(1)}%</span>
                        ${getTrendArrow(row.l5_1, row.l15_1)}
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line1}</td>
            <td>
                <div class="player-cell">
                    <span class="player-name">${row.name2}</span>
                    <div class="player-stats">
                        <span class="side-badge side-${row.side2}">${row.side2}</span>
                        <span class="hit-rate ${getHitRateClass(row.hitRate2)} tooltip" data-tooltip="L-5: ${(row.l5_2*100).toFixed(0)}% | L-15: ${(row.l15_2*100).toFixed(0)}%">${row.hitRate2.toFixed(1)}%</span>
                        ${getTrendArrow(row.l5_2, row.l15_2)}
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line2}</td>
            <td>
                <div class="player-cell">
                    <span class="player-name">${row.name3}</span>
                    <div class="player-stats">
                        <span class="side-badge side-${row.side3}">${row.side3}</span>
                        <span class="hit-rate ${getHitRateClass(row.hitRate3)} tooltip" data-tooltip="L-5: ${(row.l5_3*100).toFixed(0)}% | L-15: ${(row.l15_3*100).toFixed(0)}%">${row.hitRate3.toFixed(1)}%</span>
                        ${getTrendArrow(row.l5_3, row.l15_3)}
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line3}</td>
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
        const avgEV = (data.reduce((sum, row) => sum + row.ev, 0) / data.length).toFixed(2);
        const maxEV = Math.max(...data.map(row => row.ev)).toFixed(2);
        const avgROI = (data.reduce((sum, row) => sum + row.roi, 0) / data.length).toFixed(1);
        
        statsHTML = `
            <div class="stat-card">
                <div class="stat-label">Average EV</div>
                <div class="stat-value">${avgEV}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max EV</div>
                <div class="stat-value">${maxEV}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg ROI</div>
                <div class="stat-value">${avgROI}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Picks</div>
                <div class="stat-value">${data.length}</div>
            </div>
        `;
    } else if (currentType === 'hitrates') {
        const avgOverPct = (data.reduce((sum, row) => sum + row.overPct, 0) / data.length * 100).toFixed(1);
        const avgL10 = (data.reduce((sum, row) => sum + row.l10, 0) / data.length * 100).toFixed(1);
        const hotPlayers = data.filter(row => row.l5 >= 0.7).length;
        const trendingUp = data.filter(row => (row.l5 - row.l15) > 0.15).length;
        
        statsHTML = `
            <div class="stat-card">
                <div class="stat-label">Avg Over Hit Rate</div>
                <div class="stat-value">${avgOverPct}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg L-10 Rate</div>
                <div class="stat-value">${avgL10}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Hot Players (L-5 ≥70%)</div>
                <div class="stat-value">${hotPlayers}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Trending Up ↑</div>
                <div class="stat-value">${trendingUp}</div>
            </div>
        `;
    } else {
        const avgEV = (data.reduce((sum, row) => sum + row.ev, 0) / data.length).toFixed(2);
        const maxEV = Math.max(...data.map(row => row.ev)).toFixed(2);
        const avgKelly = (data.reduce((sum, row) => sum + row.kelly, 0) / data.length * 100).toFixed(1);
        const avgHitRate = currentType === 'pairs' 
            ? ((data.reduce((sum, row) => sum + row.hitRate1 + row.hitRate2, 0) / (data.length * 2)).toFixed(1))
            : ((data.reduce((sum, row) => sum + row.hitRate1 + row.hitRate2 + row.hitRate3, 0) / (data.length * 3)).toFixed(1));
        
        statsHTML = `
            <div class="stat-card">
                <div class="stat-label">Average EV</div>
                <div class="stat-value">${avgEV}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max EV</div>
                <div class="stat-value">${maxEV}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Hit Rate</div>
                <div class="stat-value">${avgHitRate}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Picks</div>
                <div class="stat-value">${data.length}</div>
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
        } else {
            if (currentPropType === 'points') return underdogPointsHitRates;
            if (currentPropType === 'assists') return underdogAssistsHitRates;
            if (currentPropType === 'rebounds') return underdogReboundsHitRates;
            if (currentPropType === 'blocks') return underdogBlocksHitRates;
            if (currentPropType === 'steals') return underdogStealsHitRates;
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

