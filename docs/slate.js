var SLATE_PRIZEPICKS = [];
var SLATE_UNDERDOG = [];
var SLATE_DRAFTKINGS = [];
var SLATE_BETR = [];
var activeBook = "prizepicks";
var activeView = "pairs";
var ALL_LINE_PROBS = [];
var lineProbsState = "idle";
/** null = all stats; otherwise MARKET key e.g. PTS, AST, REB */
var activePlayerStat = null;

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

function parseJsonl(text) {
  var lines = String(text).split(/\r?\n/);
  var out = [];
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i].trim();
    if (!line) continue;
    try {
      out.push(JSON.parse(line));
    } catch (e) {
      /* skip bad lines */
    }
  }
  return out;
}

function fetchLineProbsWithFallback(urls) {
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
          return res.text();
        })
        .then(function (text) {
          resolve(parseJsonl(text));
        })
        .catch(function () {
          tryNext();
        });
    }
    tryNext();
  });
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
    fetchSlateWithFallback(jsonUrls("underdog.json")),
    fetchSlateWithFallback(jsonUrls("draftKings.json")),
    fetchSlateWithFallback(jsonUrls("betr.json"))
  ])
    .then(function (results) {
      SLATE_PRIZEPICKS = sortByEvDesc(results[0]);
      SLATE_UNDERDOG = sortByEvDesc(results[1]);
      SLATE_DRAFTKINGS = sortByEvDesc(results[2]);
      SLATE_BETR = sortByEvDesc(results[3]);
      if (
        !SLATE_PRIZEPICKS.length &&
        !SLATE_UNDERDOG.length &&
        !SLATE_DRAFTKINGS.length &&
        !SLATE_BETR.length
      ) {
        cards.innerHTML =
          '<p class="load-msg load-err">Could not load slate JSON. Serve the site over HTTP and place <code>prizepicks.json</code>, <code>underdog.json</code>, <code>draftKings.json</code>, and <code>betr.json</code> under <code>data/props/ev_analysis/</code>.</p>';
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

function bookDisplayLabel(book) {
  switch (book) {
    case "underdog":
      return "Underdog";
    case "draftkings":
      return "DraftKings Pick 6";
    case "betr":
      return "Betr";
    default:
      return "PrizePicks";
  }
}

function activeSlateJsonFile() {
  switch (activeBook) {
    case "underdog":
      return "underdog.json";
    case "draftkings":
      return "draftKings.json";
    case "betr":
      return "betr.json";
    default:
      return "prizepicks.json";
  }
}

function initBookToggle() {
  var root = document.getElementById("bookDropdown");
  var trigger = document.getElementById("bookDropdownTrigger");
  var menu = document.getElementById("bookDropdownMenu");
  var valueEl = document.getElementById("bookDropdownValue");
  var opts = document.querySelectorAll(".book-dropdown-option");
  if (!root || !trigger || !menu || !valueEl) return;

  function syncUi() {
    valueEl.textContent = bookDisplayLabel(activeBook);
    for (var i = 0; i < opts.length; i++) {
      var b = opts[i].getAttribute("data-book");
      var on = b === activeBook;
      opts[i].setAttribute("aria-selected", on ? "true" : "false");
      opts[i].classList.toggle("book-dropdown-option--current", on);
    }
  }

  function closeMenu() {
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
    trigger.classList.remove("book-dropdown-trigger--open");
  }

  function openMenu() {
    menu.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
    trigger.classList.add("book-dropdown-trigger--open");
  }

  function toggleMenu() {
    if (menu.hidden) openMenu();
    else closeMenu();
  }

  syncUi();

  trigger.addEventListener("click", function (e) {
    e.stopPropagation();
    toggleMenu();
  });

  for (var j = 0; j < opts.length; j++) {
    opts[j].addEventListener("click", function () {
      var book = this.getAttribute("data-book");
      if (!book) return;
      if (book !== activeBook) {
        activeBook = book;
        syncUi();
        if (activeView === "pairs") render();
      }
      closeMenu();
    });
  }

  root.addEventListener("click", function (e) {
    e.stopPropagation();
  });

  document.addEventListener("click", function () {
    if (!menu.hidden) closeMenu();
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !menu.hidden) {
      closeMenu();
      trigger.focus();
    }
  });
}

function updateSlateNavAria() {
  var tabs = document.querySelectorAll("[data-view]");
  for (var i = 0; i < tabs.length; i++) {
    var t = tabs[i];
    if (t.getAttribute("data-view") === activeView) {
      t.setAttribute("aria-current", "page");
    } else {
      t.removeAttribute("aria-current");
    }
  }
}

function setActiveView(view) {
  if (!view || view === activeView) return;
  activeView = view;
  updateSlateNavAria();
  var secPairs = document.getElementById("sectionPairs");
  var secPlayers = document.getElementById("sectionPlayers");
  if (secPairs) secPairs.hidden = activeView !== "pairs";
  if (secPlayers) secPlayers.hidden = activeView !== "players";
  if (activeView === "pairs") {
    render();
  } else {
    loadLineProbs();
  }
}

function initMainNav() {
  var tabs = document.querySelectorAll("[data-view]");
  for (var i = 0; i < tabs.length; i++) {
    tabs[i].addEventListener("click", function () {
      var v = this.getAttribute("data-view");
      if (v) setActiveView(v);
    });
  }
}

function loadLineProbs() {
  var panel = document.getElementById("playersPanel");
  if (lineProbsState === "loading") return;
  if (lineProbsState === "loaded") {
    renderPlayersPanel();
    return;
  }
  lineProbsState = "loading";
  if (panel) panel.innerHTML = '<p class="load-msg">Loading player lines…</p>';
  fetchLineProbsWithFallback(jsonUrls("all_line_probs.json")).then(function (rows) {
    ALL_LINE_PROBS = rows;
    lineProbsState = "loaded";
    renderPlayersPanel();
  });
}

function fmtSignedEv(x) {
  var n = Number(x);
  var sign = n >= 0 ? "+" : "-";
  return sign + fmtEv(Math.abs(n));
}

/** Historical hit rate for the same side as Best (over uses OVER_RATE_L*; under uses 1 − that rate). */
function fmtBestSideHitPct(overRate, bestIsOver) {
  if (overRate == null || overRate === "") return "—";
  var n = Number(overRate);
  if (isNaN(n)) return "—";
  var p = bestIsOver ? n : 1 - n;
  return Math.round(p * 100) + "%";
}

function filterPlayerRows(rows, query) {
  var q = String(query || "")
    .trim()
    .toLowerCase();
  if (!q) return rows.slice();
  var parts = q.split(/\s+/).filter(Boolean);
  return rows.filter(function (r) {
    var name = String(r.PLAYER_NAME || "").toLowerCase();
    for (var i = 0; i < parts.length; i++) {
      if (name.indexOf(parts[i]) === -1) return false;
    }
    return true;
  });
}

function filterPlayerRowsByStat(rows, stat) {
  if (!stat) return rows.slice();
  var want = String(stat).toUpperCase();
  return rows.filter(function (r) {
    var m = String(r.MARKET || r.CATEGORY || "").toUpperCase();
    return m === want;
  });
}

function bookPillClass(book) {
  var b = String(book || "").toLowerCase();
  if (b.indexOf("prize") !== -1) return "book-prizepicks";
  if (b.indexOf("underdog") !== -1) return "book-underdog";
  if (b.indexOf("draft") !== -1) return "book-draftkings";
  if (b.indexOf("betr") !== -1) return "book-betr";
  return "book-unknown";
}

function htmlBookPill(book) {
  var label = String(book || "").trim() || "—";
  return (
    '<span class="book-pill ' + bookPillClass(book) + '">' + escapeHtml(label) + "</span>"
  );
}

function renderPlayersPanel() {
  var el = document.getElementById("playersPanel");
  if (!el) return;
  if (!ALL_LINE_PROBS.length) {
    el.innerHTML =
      '<p class="load-msg load-err">No rows in <code>all_line_probs.json</code> or file not found. Export JSONL to <code>data/props/ev_analysis/all_line_probs.json</code>.</p>';
    return;
  }
  var searchEl = document.getElementById("playerSearch");
  var q = searchEl ? searchEl.value : "";
  var sorted = ALL_LINE_PROBS.slice().sort(function (a, b) {
    var ba = Math.max(Number(a.EV_OVER), Number(a.EV_UNDER));
    var bb = Math.max(Number(b.EV_OVER), Number(b.EV_UNDER));
    if (bb !== ba) return bb - ba;
    var cmp = String(a.PLAYER_NAME || "").localeCompare(String(b.PLAYER_NAME || ""));
    if (cmp) return cmp;
    cmp = String(a.MARKET || a.CATEGORY || "").localeCompare(String(b.MARKET || b.CATEGORY || ""));
    if (cmp) return cmp;
    if (Number(a.LINE) !== Number(b.LINE)) return Number(a.LINE) - Number(b.LINE);
    return String(a.LINE_BOOKMAKER || "").localeCompare(String(b.LINE_BOOKMAKER || ""));
  });
  var byStat = filterPlayerRowsByStat(sorted, activePlayerStat);
  var filtered = filterPlayerRows(byStat, q);
  var html =
    '<div class="players-wrap"><table class="players-table"><thead><tr>' +
    "<th>Player</th><th>Book</th><th>Mkt</th>" +
    '<th class="num">Line</th><th class="num">Proj</th>' +
    '<th class="num">O</th><th class="num">U</th>' +
    '<th class="num">EV O</th><th class="num">EV U</th>' +
    "<th>Best</th>" +
    '<th class="num" title="Hit rate on best side (last 5 games)">L5</th>' +
    '<th class="num" title="Hit rate on best side (last 10 games)">L10</th>' +
    '<th class="num" title="Hit rate on best side (last 15 games)">L15</th>' +
    "</tr></thead><tbody>";
  if (!filtered.length) {
    html +=
      '<tr><td colspan="13" class="players-empty">No rows match your search.</td></tr>';
  }
  for (var i = 0; i < filtered.length; i++) {
    var r = filtered[i];
    var oEv = Number(r.EV_OVER);
    var uEv = Number(r.EV_UNDER);
    var bestOver = oEv >= uEv;
    var bestSide = bestOver ? "over" : "under";
    html +=
      "<tr>" +
      "<td>" + escapeHtml(r.PLAYER_NAME || "") + "</td>" +
      "<td>" + htmlBookPill(r.LINE_BOOKMAKER) + "</td>" +
      "<td>" + escapeHtml(r.MARKET || r.CATEGORY || "") + "</td>" +
      '<td class="num">' + fmt1(r.LINE) + "</td>" +
      '<td class="num">' + fmt1(r.STAT_Q50) + "</td>" +
      '<td class="num">' + (r.ODDS_OVER != null ? String(r.ODDS_OVER) : "—") + "</td>" +
      '<td class="num">' + (r.ODDS_UNDER != null ? String(r.ODDS_UNDER) : "—") + "</td>" +
      '<td class="num ' + (bestOver ? "ev-best" : "ev-muted") + '">' + fmtSignedEv(oEv) + "%</td>" +
      '<td class="num ' + (!bestOver ? "ev-best" : "ev-muted") + '">' + fmtSignedEv(uEv) + "%</td>" +
      "<td>" +
      '<span class="side-tag ' + bestSide + '">' + escapeHtml(bestSide.toUpperCase()) + "</span>" +
      "</td>" +
      '<td class="num">' + fmtBestSideHitPct(r.OVER_RATE_L5, bestOver) + "</td>" +
      '<td class="num">' + fmtBestSideHitPct(r.OVER_RATE_L10, bestOver) + "</td>" +
      '<td class="num">' + fmtBestSideHitPct(r.OVER_RATE_L15, bestOver) + "</td>" +
      "</tr>";
  }
  html += "</tbody></table></div>";
  el.innerHTML = html;
}

function initPlayerSearch() {
  var input = document.getElementById("playerSearch");
  if (!input) return;
  input.addEventListener("input", function () {
    if (lineProbsState === "loaded") renderPlayersPanel();
  });
}

function syncStatPillUi() {
  var pills = document.querySelectorAll(".stat-pill");
  for (var i = 0; i < pills.length; i++) {
    var s = pills[i].getAttribute("data-stat");
    var on = s === "ALL" ? !activePlayerStat : s === activePlayerStat;
    pills[i].classList.toggle("active", on);
    pills[i].setAttribute("aria-pressed", on ? "true" : "false");
  }
}

function initPlayerStatFilter() {
  var bar = document.querySelector(".stat-filter");
  if (!bar) return;
  bar.addEventListener("click", function (e) {
    var t = e.target && e.target.closest(".stat-pill");
    if (!t) return;
    var s = t.getAttribute("data-stat");
    if (s === "ALL") {
      activePlayerStat = null;
    } else {
      activePlayerStat = activePlayerStat === s ? null : s;
    }
    syncStatPillUi();
    if (lineProbsState === "loaded") renderPlayersPanel();
  });
}

function currentSlate() {
  switch (activeBook) {
    case "underdog":
      return SLATE_UNDERDOG;
    case "draftkings":
      return SLATE_DRAFTKINGS;
    case "betr":
      return SLATE_BETR;
    default:
      return SLATE_PRIZEPICKS;
  }
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
      '<p class="player-name">' +
        escapeHtml(name) +
        ' <span class="player-team">- ' + escapeHtml(team) + "</span>" +
      "</p>" +
      '<p class="subtitle">vs ' + escapeHtml(opponent) + "</p>" +
      '<div class="line-row">' +
        '<span class="line-num">' + fmt1(line) + "</span>" +
        '<span class="market-lbl">' + escapeHtml(market) + "</span>" +
        '<span class="side-pill ' + sideClass + '">' + escapeHtml(String(side).toUpperCase()) + "</span>" +
      "</div>" +
      '<p class="model-line">Model predicts ' + fmt1(prediction) +
      ' <span class="' + diffClass + '">(' + diffDisplay(side, prediction, line) + ")</span></p>" +
      '<div class="mini-grid">' +
        "<span>L10 Stat Avg.</span><span>" + fmt1(avgL10) + "</span>" +
        "<span>vs matchup</span><span>" + fmt1(avgVs) + " (" + matchupGames + " games)</span>" +
        "<span>Minutes Avg.</span><span>" + fmt1(avgMin) + "</span>" +
        "<span>Opp Def Rank</span><span>" + ordSuffix(defRank) + "</span>" +
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
  if (activeView !== "pairs") return;
  const sorted = currentSlate();
  const el = document.getElementById("cards");
  if (!sorted.length) {
    var label = bookDisplayLabel(activeBook);
    el.innerHTML =
      '<p class="load-msg">No parlays in the ' + label + " slate. Export <code>" +
      activeSlateJsonFile() +
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

initMainNav();
initPlayerSearch();
initPlayerStatFilter();
try {
  var q = new URLSearchParams(window.location.search);
  var viewParam = q.get("view");
  if (viewParam === "players" || viewParam === "pairs") {
    setActiveView(viewParam);
  }
} catch (e) {
  /* ignore */
}
loadSlates();
