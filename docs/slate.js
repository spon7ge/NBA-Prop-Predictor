var SLATE_PRIZEPICKS = [];
var SLATE_UNDERDOG = [];
var activeBook = "prizepicks";

function jsonUrls(filename) {
  var a = "data/props/ev_analysis/" + filename;
  var b = "../data/props/ev_analysis/" + filename;
  var path = typeof window !== "undefined" && window.location && window.location.pathname
    ? window.location.pathname
    : "";
  // When the page lives under /docs/, prefer repo-root data/ first so PrizePicks/Underdog
  // always come from data/props/ev_analysis/ (authoritative). The copy under docs/data/ is fallback.
  if (path.indexOf("/docs/") !== -1) {
    return [b, a];
  }
  return [a, b];
}

function fetchSlateWithFallback(urls) {
  return new Promise(function (resolve) {
    var i = 0;
    function tryNext() {
      if (i >= urls.length) {
        resolve([]);
        return;
      }
      var url = urls[i++];
      fetch(url, { cache: "no-store" })
        .then(function (res) {
          if (!res.ok) throw new Error("HTTP " + res.status);
          return res.json();
        })
        .then(function (data) {
          resolve(Array.isArray(data) ? data : []);
        })
        .catch(function () {
          tryNext();
        });
    }
    tryNext();
  });
}

function sortByEvDesc(arr) {
  return arr.slice().sort(function (a, b) {
    return Number(b["EV"]) - Number(a["EV"]);
  });
}

function loadSlates() {
  var cards = document.getElementById("cards");
  cards.innerHTML = '<p class="load-msg">Loading slates…</p>';

  Promise.all([
    fetchSlateWithFallback(jsonUrls("prizepicks.json")),
    fetchSlateWithFallback(jsonUrls("underdog.json"))
  ])
    .then(function (results) {
      SLATE_PRIZEPICKS = sortByEvDesc(results[0]);
      SLATE_UNDERDOG = sortByEvDesc(results[1]);
      if (!SLATE_PRIZEPICKS.length && !SLATE_UNDERDOG.length) {
        cards.innerHTML =
          '<p class="load-msg load-err">Could not load <code>prizepicks.json</code> or <code>underdog.json</code>. Serve the site over HTTP and place both files under <code>data/props/ev_analysis/</code>.</p>';
        return;
      }
      initBookToggle();
      render();
    })
    .catch(function () {
      cards.innerHTML =
        '<p class="load-msg load-err">Failed to load slates. Check the console and ensure JSON paths are reachable.</p>';
    });
}

function initBookToggle() {
  var btns = document.querySelectorAll(".book-btn");
  for (var j = 0; j < btns.length; j++) {
    btns[j].addEventListener("click", function () {
      var book = this.getAttribute("data-book");
      if (!book || book === activeBook) return;
      activeBook = book;
      for (var k = 0; k < btns.length; k++) {
        btns[k].classList.toggle("active", btns[k].getAttribute("data-book") === activeBook);
      }
      render();
    });
  }
}

function currentSlate() {
  return activeBook === "underdog" ? SLATE_UNDERDOG : SLATE_PRIZEPICKS;
}

function ordSuffix(n) {
  const i = Math.floor(Math.abs(n));
  const j = i % 100;
  if (j >= 11 && j <= 13) return i + "th";
  switch (i % 10) {
    case 1: return i + "st";
    case 2: return i + "nd";
    case 3: return i + "rd";
    default: return i + "th";
  }
}

function spreadFmt(n) {
  const v = Number(n);
  const sign = v > 0 ? "+" : "";
  return sign + v.toFixed(1);
}

function hitRate(d, n) {
  const r = Number(d);
  const s = String(n).toLowerCase();
  return s === "over" ? r : 1 - r;
}

function pct0(x) {
  return Math.round(Number(x) * 100);
}

function fmt1(x) {
  return Number(x).toFixed(1);
}

function fmtEv(x) {
  return Number(x).toFixed(2);
}

function modelDiffGood(side, prediction, line) {
  const d = Number(prediction) - Number(line);
  const s = String(side).toLowerCase();
  if (s === "over") return d > 0;
  return d < 0;
}

function diffDisplay(side, prediction, line) {
  const d = Number(prediction) - Number(line);
  const sign = d > 0 ? "+" : "";
  return sign + fmt1(d);
}

function renderLeg(leg) {
  const name = leg.name;
  const team = leg.team;
  const market = leg.market;
  const line = leg.line;
  const side = leg.side;
  const prediction = leg.prediction;
  const opponent = leg.opponent;
  const defRank = leg.defRank;
  const avgL10 = leg.avgStatL10;
  const avgMin = leg.avgMinL10;
  const avgVs = leg.avgVsMatchup;
  const matchupGames = leg.matchupGames;
  const overRate = leg.overRateL10;

  const isOver = String(side).toLowerCase() === "over";
  const sideClass = isOver ? "side-over" : "side-under";
  const hr = hitRate(overRate, side);
  const hrPct = pct0(hr);
  const diffGood = modelDiffGood(side, prediction, line);
  const diffClass = diffGood ? "diff-pos" : "diff-neg";

  return (
    '<div class="leg">' +
      '<p class="player-name">' + escapeHtml(name) + "</p>" +
      '<p class="subtitle">vs ' + escapeHtml(opponent) + " · " + escapeHtml(team) + "</p>" +
      '<div class="line-row">' +
        '<span class="line-num">' + fmt1(line) + "</span>" +
        '<span class="market-lbl">' + escapeHtml(market) + "</span>" +
        '<span class="side-pill ' + sideClass + '">' + escapeHtml(String(side).toUpperCase()) + "</span>" +
      "</div>" +
      '<p class="model-line">Model predicts ' + fmt1(prediction) +
      ' <span class="' + diffClass + '">(' + diffDisplay(side, prediction, line) + ")</span></p>" +
      '<div class="mini-grid">' +
        "<span>L10 avg</span><span>" + fmt1(avgL10) + "</span>" +
        "<span>vs matchup</span><span>" + fmt1(avgVs) + " (" + matchupGames + " games)</span>" +
        "<span>Minutes</span><span>" + fmt1(avgMin) + "</span>" +
        "<span>Opp def rank</span><span>" + ordSuffix(defRank) + "</span>" +
      "</div>" +
      '<div class="hit-wrap">' +
        '<div class="hit-label-row">' +
          "<span>Hit rate L10</span><span>" + hrPct + "%</span>" +
        "</div>" +
        '<div class="hit-track"><div class="hit-fill" style="width:' + hrPct + '%"></div></div>' +
      "</div>" +
    "</div>"
  );
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function mapRow(row) {
  return {
    name1: row["NAME 1"],
    team1: row["TEAM 1"],
    market1: row["MARKET 1"],
    line1: row["LINE 1"],
    side1: row["SIDE 1"],
    prediction1: row["PREDICTION 1"],
    opponent1: row["OPPONENT 1"],
    spread1: row["SPREAD 1"],
    total1: row["TOTAL 1"],
    defRank1: row["OPP_DEF_RANK 1"],
    avgStatL101: row["AVG_STAT_L10 1"],
    avgMinL101: row["AVG_MIN_L10 1"],
    avgVsMatchup1: row["AVG_STAT_VS_MATCHUP 1"],
    matchupGames1: row["MATCHUP_GAMES 1"],
    overRateL101: row["OVER_RATE_L10 1"],
    name2: row["NAME 2"],
    team2: row["TEAM 2"],
    market2: row["MARKET 2"],
    line2: row["LINE 2"],
    side2: row["SIDE 2"],
    prediction2: row["PREDICTION 2"],
    opponent2: row["OPPONENT 2"],
    spread2: row["SPREAD 2"],
    total2: row["TOTAL 2"],
    defRank2: row["OPP_DEF_RANK 2"],
    avgStatL102: row["AVG_STAT_L10 2"],
    avgMinL102: row["AVG_MIN_L10 2"],
    avgVsMatchup2: row["AVG_STAT_VS_MATCHUP 2"],
    matchupGames2: row["MATCHUP_GAMES 2"],
    overRateL102: row["OVER_RATE_L10 2"],
    parlayProb: row["PARLAY_PROB"],
    ev: row["EV"],
    kelly: row["KELLY"]
  };
}

function render() {
  const sorted = currentSlate();
  const el = document.getElementById("cards");
  if (!sorted.length) {
    var label = activeBook === "underdog" ? "Underdog" : "PrizePicks";
    el.innerHTML =
      '<p class="load-msg">No parlays in the ' + label + " slate. Export <code>" +
      (activeBook === "underdog" ? "underdog.json" : "prizepicks.json") +
      "</code> into <code>data/props/ev_analysis/</code>.</p>";
    return;
  }
  let html = "";
  for (let i = 0; i < sorted.length; i++) {
    const r = mapRow(sorted[i]);
    const rank = i + 1;
    const evNum = Number(r.ev);
    const evStr = (evNum >= 0 ? "+" : "-") + fmtEv(Math.abs(evNum));
    const probPct = pct0(r.parlayProb);
    const kellyPct = fmt1(r.kelly);

    const leg1 = renderLeg({
      name: r.name1,
      team: r.team1,
      market: r.market1,
      line: r.line1,
      side: r.side1,
      prediction: r.prediction1,
      opponent: r.opponent1,
      spread: r.spread1,
      total: r.total1,
      defRank: r.defRank1,
      avgStatL10: r.avgStatL101,
      avgMinL10: r.avgMinL101,
      avgVsMatchup: r.avgVsMatchup1,
      matchupGames: r.matchupGames1,
      overRateL10: r.overRateL101
    });
    const leg2 = renderLeg({
      name: r.name2,
      team: r.team2,
      market: r.market2,
      line: r.line2,
      side: r.side2,
      prediction: r.prediction2,
      opponent: r.opponent2,
      spread: r.spread2,
      total: r.total2,
      defRank: r.defRank2,
      avgStatL10: r.avgStatL102,
      avgMinL10: r.avgMinL102,
      avgVsMatchup: r.avgVsMatchup2,
      matchupGames: r.matchupGames2,
      overRateL10: r.overRateL102
    });

    html +=
      '<article class="card">' +
        '<div class="card-header">' +
          '<span class="rank-label">#' + rank + " pick</span>" +
          '<div class="badges">' +
            '<span class="pill pill-ev">EV ' + evStr + "%</span>" +
            '<span class="pill pill-prob">Hit prob ' + probPct + "%</span>" +
            '<span class="pill pill-kelly">Kelly ' + kellyPct + "%</span>" +
          "</div>" +
        "</div>" +
        '<div class="legs">' + leg1 + leg2 + "</div>" +
        '<div class="card-footer">' +
          "<div>Game total " + fmt1(r.total1) + "<br />Spread " + spreadFmt(r.spread1) + "</div>" +
          "<div>Game total " + fmt1(r.total2) + "<br />Spread " + spreadFmt(r.spread2) + "</div>" +
        "</div>" +
      "</article>";
  }
  el.innerHTML = html;
}

loadSlates();
