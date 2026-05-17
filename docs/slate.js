var SLATE_LEG_COUNTS = [2, 3, 5, 6];
var SLATES = {
  2: { prizepicks: [], underdog: [], draftkings: [], betr: [] },
  3: { prizepicks: [], underdog: [], draftkings: [], betr: [] },
  5: { prizepicks: [], underdog: [], draftkings: [], betr: [] },
  6: { prizepicks: [], underdog: [], draftkings: [], betr: [] }
};
var activeBook = "prizepicks";
/** Parlay size: 2, 3, 5, or 6 legs (4-leg skipped in pipeline). */
var activeLegs = 2;
var activeView = "pairs";
var ALL_ENRICHED = [];
var enrichedState = "idle";
/** null = all stats; otherwise MARKET key e.g. PTS, AST, REB */
var activePlayerStat = null;
/** null = all platforms */
var activePlatform = null;
/** null = all tiers */
var activeTier = null;
/** null = default (tier+conviction); otherwise column id for All Players table sort */
var playersSortKey = null;
var playersSortDir = "desc";
var expandedPlayer = null;
var playersGroupSort = "edge_desc";
var DFS_BREAK_EVEN = 137 / (137 + 100);

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

function enrichedUrls() {
  var urls = [];
  var path = typeof window !== "undefined" && window.location && window.location.pathname
    ? window.location.pathname : "";
  var inDocs = path.indexOf("/docs/") !== -1;
  var now = new Date();
  // Prefer stable "latest" pointer (pipeline should write this).
  // Falls back to date-based files for backwards compatibility.
  var latest = "dfs_enriched_latest.json";
  var la = "data/props/enriched/" + latest;
  var lb = "../data/props/enriched/" + latest;
  urls.push(inDocs ? lb : la);
  urls.push(inDocs ? la : lb);
  for (var i = 0; i < 7; i++) {
    var d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - i);
    var y = d.getFullYear();
    var mo = String(d.getMonth() + 1).padStart(2, "0");
    var dy = String(d.getDate()).padStart(2, "0");
    var fname = "dfs_enriched_" + y + mo + dy + ".json";
    var a = "data/props/enriched/" + fname;
    var b = "../data/props/enriched/" + fname;
    urls.push(inDocs ? b : a);
    urls.push(inDocs ? a : b);
  }
  return urls;
}

function fetchEnrichedWithFallback(urls) {
  return new Promise(function (resolve) {
    var i = 0;
    function tryNext() {
      if (i >= urls.length) { resolve([]); return; }
      var url = urls[i++];
      fetch(url, { cache: "no-store" })
        .then(function (res) {
          if (!res.ok) throw new Error("HTTP " + res.status);
          return res.json();
        })
        .then(function (data) {
          var picks = data && Array.isArray(data.picks) ? data.picks : [];
          if (picks.length) { resolve(picks); } else { tryNext(); }
        })
        .catch(function () { tryNext(); });
    }
    tryNext();
  });
}

function sortByEvDesc(arr) {
  return arr.slice().sort(function (a, b) {
    return Number(b["EV"]) - Number(a["EV"]);
  });
}

/**
 * EV slate JSON from the pipeline is nested (LEGS[]). The Top Legs UI still uses
 * mapRow() which expects flat CSV-style keys (NAME 1, LINE 1, …). Normalize here
 * so filenames and render logic stay the same.
 */
function normalizeParlayRowForUi(row) {
  if (!row || typeof row !== "object") return null;
  if (row["NAME 1"] != null && row.LEGS == null) return row;
  var legs = row.LEGS;
  if (!Array.isArray(legs) || legs.length < 2) return null;
  if (row.N_LEGS != null && legs.length !== row.N_LEGS) return null;
  var out = {
    PARLAY_PROB: row.PARLAY_PROB,
    EV: row.EV,
    KELLY: row.KELLY != null ? row.KELLY : row.KELLY_QUARTER
  };
  for (var i = 0; i < legs.length; i++) {
    var L = legs[i];
    var n = i + 1;
    var gc = L.game_context || {};
    var vs = L.vs_opp || {};
    var fo = L.form || {};
    var md = L.model || {};
    out["NAME " + n] = L.player != null ? L.player : L.display_name;
    out["TEAM " + n] = L.team_abbr != null ? L.team_abbr : L.team;
    out["MARKET " + n] = L.market;
    out["LINE " + n] = L.dfs_line;
    out["SIDE " + n] = L.side != null ? L.side : md.lean;
    out["PREDICTION " + n] = md.stat_q50;
    out["OPPONENT " + n] =
      L.opponent_abbr != null && String(L.opponent_abbr).trim() !== ""
        ? L.opponent_abbr
        : L.opponent;
    out["SPREAD " + n] = gc.spread;
    out["TOTAL " + n] = gc.game_total;
    out["OPP_DEF_RANK " + n] = gc.opp_def_rating_rank;
    out["OPP_PACE_RANK " + n] = gc.opp_pace_rank;
    out["AVG_STAT_L10 " + n] = L.avg_stat_l10;
    out["AVG_STAT_VS_MATCHUP " + n] = vs.avg_stat;
    out["MATCHUP_GAMES " + n] = vs.n_games;
    out["OVER_RATE_L10 " + n] = fo.over_l10;
  }
  return out;
}

function normalizeSlateArray(arr) {
  if (!Array.isArray(arr)) return [];
  var out = [];
  for (var i = 0; i < arr.length; i++) {
    var r = normalizeParlayRowForUi(arr[i]);
    if (r) out.push(r);
  }
  return out;
}

function slateJsonFilename(book, nLegs) {
  var bases = {
    prizepicks: "prizepicks",
    underdog: "underdog",
    draftkings: "draftKings",
    betr: "betr"
  };
  var base = bases[book] || "prizepicks";
  if (nLegs === 2) return base + ".json";
  return base + "_" + nLegs + "leg.json";
}

function loadSlates() {
  var cards = document.getElementById("cards");
  cards.innerHTML = '<p class="load-msg">Loading slates…</p>';

  var books = ["prizepicks", "underdog", "draftkings", "betr"];
  var fetches = [];
  for (var li = 0; li < SLATE_LEG_COUNTS.length; li++) {
    var nLegs = SLATE_LEG_COUNTS[li];
    for (var bi = 0; bi < books.length; bi++) {
      fetches.push(fetchSlateWithFallback(jsonUrls(slateJsonFilename(books[bi], nLegs))));
    }
  }

  Promise.all(fetches)
    .then(function (results) {
      var idx = 0;
      for (var li2 = 0; li2 < SLATE_LEG_COUNTS.length; li2++) {
        var n = SLATE_LEG_COUNTS[li2];
        for (var bi2 = 0; bi2 < books.length; bi2++) {
          SLATES[n][books[bi2]] = sortByEvDesc(normalizeSlateArray(results[idx++]));
        }
      }
      var hasAny = false;
      for (var li3 = 0; li3 < SLATE_LEG_COUNTS.length; li3++) {
        var legN = SLATE_LEG_COUNTS[li3];
        var bucket = SLATES[legN];
        if (bucket.prizepicks.length || bucket.underdog.length || bucket.draftkings.length || bucket.betr.length) {
          hasAny = true;
          break;
        }
      }
      if (!hasAny) {
        cards.innerHTML =
          '<p class="load-msg load-err">Could not load slate JSON. Serve the site over HTTP and place slate files under <code>data/props/ev_analysis/</code> (e.g. <code>prizepicks.json</code>, <code>prizepicks_3leg.json</code>, <code>prizepicks_5leg.json</code>, <code>prizepicks_6leg.json</code>).</p>';
        return;
      }
      initBookToggle();
      initLegsToggle();
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
  return slateJsonFilename(activeBook, activeLegs);
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

function legsDisplayLabel(legs) {
  var n = Number(legs);
  if (n === 3) return "3-Leg";
  if (n === 5) return "5-Leg";
  if (n === 6) return "6-Leg";
  return "2-Leg";
}

function initLegsToggle() {
  var root = document.getElementById("legsDropdown");
  var trigger = document.getElementById("legsDropdownTrigger");
  var menu = document.getElementById("legsDropdownMenu");
  var valueEl = document.getElementById("legsDropdownValue");
  var opts = document.querySelectorAll(".legs-dropdown-option");
  if (!root || !trigger || !menu || !valueEl) return;

  function syncUi() {
    valueEl.textContent = legsDisplayLabel(activeLegs);
    for (var i = 0; i < opts.length; i++) {
      var legN = parseInt(opts[i].getAttribute("data-legs"), 10);
      var on = legN === activeLegs;
      opts[i].setAttribute("aria-selected", on ? "true" : "false");
      opts[i].classList.toggle("legs-dropdown-option--current", on);
    }
  }

  function closeMenu() {
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
    trigger.classList.remove("legs-dropdown-trigger--open");
  }

  function openMenu() {
    menu.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
    trigger.classList.add("legs-dropdown-trigger--open");
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
      var raw = this.getAttribute("data-legs");
      if (raw == null) return;
      var legN = parseInt(raw, 10);
      if (isNaN(legN)) return;
      if (legN !== activeLegs) {
        activeLegs = legN;
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
  if (enrichedState === "loading") return;
  if (enrichedState === "loaded") { renderPlayersPanel(); return; }
  enrichedState = "loading";
  if (panel) panel.innerHTML = '<p class="load-msg">Loading player data…</p>';
  fetchEnrichedWithFallback(enrichedUrls()).then(function (picks) {
    var supported = { PTS: true, REB: true, AST: true };
    ALL_ENRICHED = picks.filter(function (p) { return supported[p.market]; });
    enrichedState = "loaded";
    renderPlayersPanel();
  });
}

function fmtSignedEv(x) {
  var n = Number(x);
  var sign = n >= 0 ? "+" : "-";
  return sign + fmtEv(Math.abs(n));
}

/** 0–1 probability as a percentage string. */
function fmtOverRatePct(overRate) {
  if (overRate == null || overRate === "") return "—";
  var n = Number(overRate);
  if (isNaN(n)) return "—";
  return Math.round(n * 100) + "%";
}

function tierPriority(tier) {
  if (tier === "sharp_verified") return 0;
  if (tier === "conflict") return 1;
  return 2;
}

function htmlTierPill(tier) {
  var cls, label;
  switch (String(tier || "")) {
    case "sharp_verified": cls = "tier-sharp"; label = "Verified"; break;
    case "conflict": cls = "tier-conflict"; label = "Conflict"; break;
    default: cls = "tier-nomodel"; label = "No Model"; break;
  }
  return '<span class="tier-pill ' + cls + '">' + label + "</span>";
}

function htmlLeanPill(lean) {
  if (!lean) return "—";
  var s = String(lean).toUpperCase();
  var cls = s === "OVER" ? "side-over" : "side-under";
  return '<span class="side-pill ' + cls + '">' + escapeHtml(s) + "</span>";
}

function filterPlayerRows(rows, query) {
  var q = String(query || "").trim().toLowerCase();
  if (!q) return rows.slice();
  var parts = q.split(/\s+/).filter(Boolean);
  return rows.filter(function (r) {
    var name = String(r.player || r.display_name || "").toLowerCase();
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
    return String(r.market || "").toUpperCase() === want;
  });
}

function filterByPlatform(rows, platform) {
  if (!platform) return rows.slice();
  return rows.filter(function (r) { return String(r.platform || "") === platform; });
}

function filterByTier(rows, tier) {
  if (!tier) return rows.slice();
  return rows.filter(function (r) { return String(r.tier || "") === tier; });
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

function parseSortNumber(val) {
  if (val == null || val === "") return null;
  var n = Number(val);
  if (isNaN(n)) return null;
  return n;
}

function defaultSortPlayersRows(rows) {
  return rows.slice().sort(function (a, b) {
    var ta = tierPriority(a.tier), tb = tierPriority(b.tier);
    if (ta !== tb) return ta - tb;
    var ma = a.model ? Math.abs(Number(a.model.p_over || 0.5) - 0.5) : 0;
    var mb = b.model ? Math.abs(Number(b.model.p_over || 0.5) - 0.5) : 0;
    if (mb !== ma) return mb - ma;
    return String(a.player || "").localeCompare(String(b.player || ""));
  });
}

function comparePlayersColumn(a, b, key, dir) {
  function strCmp(get) {
    var sa = String(get(a) || "");
    var sb = String(get(b) || "");
    return dir === "desc"
      ? sb.localeCompare(sa, undefined, { sensitivity: "base" })
      : sa.localeCompare(sb, undefined, { sensitivity: "base" });
  }
  function numCmp(get) {
    var va = parseSortNumber(get(a));
    var vb = parseSortNumber(get(b));
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    return dir === "desc" ? vb - va : va - vb;
  }
  switch (key) {
    case "player":   return strCmp(function (r) { return r.player || r.display_name; });
    case "platform": return strCmp(function (r) { return r.platform; });
    case "mkt":      return strCmp(function (r) { return r.market; });
    case "line":     return numCmp(function (r) { return r.dfs_line; });
    case "tier":
      var ta = tierPriority(a.tier), tb2 = tierPriority(b.tier);
      return dir === "desc" ? ta - tb2 : tb2 - ta;
    case "modelProb":     return numCmp(function (r) { return r.model && r.model.p_over; });
    case "sharpProb":     return numCmp(function (r) { return r.sharp && r.sharp.no_vig_over; });
    case "consensusProb": return numCmp(function (r) { return r.consensus && r.consensus.mean_no_vig_over_same_line; });
    case "statProj":      return numCmp(function (r) { return r.model && r.model.stat_q50; });
    case "minProj":       return numCmp(function (r) { return r.model && r.model.min_q50; });
    case "l5":            return numCmp(function (r) { return r.form && r.form.over_l5; });
    case "l10":           return numCmp(function (r) { return r.form && r.form.over_l10; });
    case "l15":           return numCmp(function (r) { return r.form && r.form.over_l15; });
    case "vsOppAvg":      return numCmp(function (r) { return r.vs_opp && r.vs_opp.avg_stat; });
    case "oppDefRank":    return numCmp(function (r) { return r.game_context && r.game_context.opp_def_rating_rank; });
    default: return 0;
  }
}

function sortPlayersRows(rows) {
  if (!playersSortKey) return defaultSortPlayersRows(rows);
  return rows.slice().sort(function (a, b) {
    return comparePlayersColumn(a, b, playersSortKey, playersSortDir);
  });
}

function playerSortTh(key, opts) {
  opts = opts || {};
  var cls = opts.className || "";
  var active = playersSortKey === key;
  var thClass = "players-sort-th" + (cls ? " " + cls : "") + (active ? " players-sort-th--active" : "");
  var ariaSort = active ? (playersSortDir === "desc" ? "descending" : "ascending") : "none";
  var title = opts.title ? ' title="' + escapeHtml(opts.title) + '"' : "";
  var arrow = active ? (playersSortDir === "desc" ? " \u2193" : " \u2191") : "";
  return (
    '<th class="' +
    thClass +
    '"' +
    title +
    ' aria-sort="' +
    ariaSort +
    '">' +
    '<button type="button" class="players-sort-btn" data-players-sort="' +
    key +
    '">' +
    escapeHtml(opts.label) +
    arrow +
    "</button></th>"
  );
}

function pickEdge(pick) {
  var m = pick && pick.model;
  if (!m) return null;
  var pOver = Number(m.p_over);
  if (isNaN(pOver)) return null;
  var pUnder = 1 - pOver;
  var bestSide, bestP;
  if (pOver >= pUnder) { bestSide = "OVER"; bestP = pOver; }
  else { bestSide = "UNDER"; bestP = pUnder; }
  return { side: bestSide, prob: bestP, edge: bestP - DFS_BREAK_EVEN };
}

function edgeBucketClass(edge) {
  if (edge == null) return "edge-neg";
  if (edge >= 0.15) return "edge-strong";
  if (edge >= 0.05) return "edge-pos";
  if (edge >= 0) return "edge-marginal";
  return "edge-neg";
}

function htmlEdgeCell(pick) {
  var e = pickEdge(pick);
  if (!e) return '<span class="edge-cell edge-neg">—</span>';
  var sign = e.edge >= 0 ? "+" : "";
  return '<span class="edge-cell ' + edgeBucketClass(e.edge) + '">' +
         sign + (e.edge * 100).toFixed(1) + '%</span>';
}

function htmlBestSidePill(pick) {
  var e = pickEdge(pick);
  if (!e) return "";
  var cls = e.side === "OVER" ? "side-over" : "side-under";
  return '<span class="side-pill ' + cls + '">' + e.side + '</span>';
}

function aggregateEnrichedByPlayer(rows) {
  var map = {};
  var order = [];
  for (var i = 0; i < rows.length; i++) {
    var r = rows[i];
    var key = r.player || r.display_name || "";
    if (!map[key]) {
      map[key] = {
        player: key,
        displayName: r.display_name || r.player || "",
        team: r.team_abbr || "",
        opp: r.opponent_abbr || null,
        is_home: r.is_home,
        picks: [],
        tiers: { sharp_verified: 0, conflict: 0, no_model: 0 },
        markets: {},
        platforms: {},
        bestEdge: null,
        bestEdgeSide: null,
      };
      order.push(key);
    }
    var agg = map[key];
    agg.picks.push(r);
    agg.markets[r.market] = true;
    agg.platforms[r.platform] = true;
    var tierKey = r.tier || "no_model";
    if (agg.tiers[tierKey] != null) agg.tiers[tierKey]++;
    var e = pickEdge(r);
    if (e && (agg.bestEdge == null || e.edge > agg.bestEdge)) {
      agg.bestEdge = e.edge;
      agg.bestEdgeSide = e.side;
    }
  }
  return order.map(function (k) { return map[k]; });
}

function sortPlayerGroups(players) {
  function n(v) { return v == null ? -Infinity : v; }
  switch (playersGroupSort) {
    case "props_desc":
      return players.slice().sort(function (a, b) {
        return b.picks.length - a.picks.length;
      });
    case "verified_desc":
      return players.slice().sort(function (a, b) {
        return (b.tiers.sharp_verified - a.tiers.sharp_verified) ||
               (b.picks.length - a.picks.length);
      });
    case "name_asc":
      return players.slice().sort(function (a, b) {
        return String(a.player).localeCompare(String(b.player));
      });
    case "edge_desc":
    default:
      return players.slice().sort(function (a, b) {
        return n(b.bestEdge) - n(a.bestEdge);
      });
  }
}

function sortGroupBtnHtml(key, label) {
  var pressed = playersGroupSort === key ? "true" : "false";
  return '<button type="button" data-grp-sort="' + key + '" aria-pressed="' + pressed + '">' +
         label + '</button>';
}

function htmlPlayerSummaryRow(p) {
  var totalProps = p.picks.length;
  var nMarkets = Object.keys(p.markets).length;
  var nPlatforms = Object.keys(p.platforms).length;
  var t = p.tiers;
  var matchupLabel = p.opp
    ? ((p.is_home ? "vs " : "@ ") + p.opp)
    : "no opp";

  var tierMix = '<span class="tier-mix" title="' +
    t.sharp_verified + ' verified \xb7 ' +
    t.conflict + ' conflict \xb7 ' +
    t.no_model + ' no model">';
  if (t.sharp_verified) tierMix += '<span class="tier-mix-seg tier-mix-sharp" style="flex:' + t.sharp_verified + '"></span>';
  if (t.conflict)       tierMix += '<span class="tier-mix-seg tier-mix-conflict" style="flex:' + t.conflict + '"></span>';
  if (t.no_model)       tierMix += '<span class="tier-mix-seg tier-mix-nomodel" style="flex:' + t.no_model + '"></span>';
  tierMix += '</span>';

  var edgeHtml;
  if (p.bestEdge == null) {
    edgeHtml = '<span class="edge-cell edge-neg">—</span>';
  } else {
    var sign = p.bestEdge >= 0 ? "+" : "";
    edgeHtml = '<span class="edge-cell ' + edgeBucketClass(p.bestEdge) + '">' +
               sign + (p.bestEdge * 100).toFixed(1) + '%</span>';
  }
  var sideHtml = p.bestEdgeSide
    ? '<span class="side-pill side-' + p.bestEdgeSide.toLowerCase() + '">' + p.bestEdgeSide + '</span>'
    : "";

  return (
    '<button type="button" class="player-row" data-player="' + escapeHtml(p.player) + '" aria-expanded="false">' +
      '<div class="player-row-main">' +
        '<span class="player-row-name">' + escapeHtml(p.displayName) + '</span>' +
        '<span class="player-row-team">' + escapeHtml(p.team) + ' \xb7 ' + escapeHtml(matchupLabel) + '</span>' +
      '</div>' +
      '<div class="player-row-stats">' +
        '<span class="player-row-stat"><b>' + totalProps + '</b> ' +
          (totalProps === 1 ? "prop" : "props") +
          ' \xb7 <span class="dim">' + nMarkets + ' mkt \xb7 ' + nPlatforms + ' bk</span></span>' +
        tierMix +
        '<span class="player-row-best">' + edgeHtml + ' ' + sideHtml + '</span>' +
      '</div>' +
      '<span class="player-row-chevron" aria-hidden="true"></span>' +
    '</button>'
  );
}

function htmlPlayerExpandedBody(picks) {
  var sorted = picks.slice();
  if (playersSortKey) {
    sorted.sort(function (a, b) { return comparePlayersColumn(a, b, playersSortKey, playersSortDir); });
  } else {
    sorted.sort(function (a, b) {
      var ea = pickEdge(a), eb = pickEdge(b);
      var va = ea ? ea.edge : -Infinity;
      var vb = eb ? eb.edge : -Infinity;
      return vb - va;
    });
  }

  var html =
    '<div class="players-wrap"><table class="players-table players-table--sortable"><thead><tr>' +
    playerSortTh("platform", { label: "Book" }) +
    playerSortTh("mkt", { label: "Mkt" }) +
    playerSortTh("line", { className: "num", label: "Line" }) +
    '<th class="num">Edge</th>' +
    playerSortTh("modelProb", { className: "num", label: "Model" }) +
    playerSortTh("sharpProb", { className: "num", label: "Sharp" }) +
    playerSortTh("consensusProb", { className: "num col-secondary", label: "Cons" }) +
    playerSortTh("statProj", { className: "num col-secondary", label: "Proj" }) +
    playerSortTh("minProj", { className: "num col-secondary", label: "Min" }) +
    playerSortTh("l5", { className: "num", label: "L5" }) +
    playerSortTh("l10", { className: "num", label: "L10" }) +
    playerSortTh("l15", { className: "num col-secondary", label: "L15" }) +
    playerSortTh("vsOppAvg", { className: "num col-secondary", label: "vs Opp" }) +
    playerSortTh("oppDefRank", { className: "num col-secondary", label: "Def Rnk" }) +
    '<th>Tier</th>' +
    "</tr></thead><tbody>";

  for (var i = 0; i < sorted.length; i++) {
    var r = sorted[i];
    var model = r.model || {};
    var sharp = r.sharp || {};
    var consensus = r.consensus || {};
    var gc = r.game_context || {};
    var form = r.form || {};
    var vsOpp = r.vs_opp || {};

    var modelCell = model.lean
      ? htmlLeanPill(model.lean) + " " + fmtOverRatePct(model.p_over)
      : "—";
    var sharpCell = sharp.lean
      ? htmlLeanPill(sharp.lean) + " " + fmtOverRatePct(sharp.no_vig_over)
      : "—";
    var consensusCell = consensus.mean_no_vig_over_same_line != null
      ? fmtOverRatePct(consensus.mean_no_vig_over_same_line) +
        ' <span class="enriched-dim">(' + (consensus.n_books_same_line || 0) + " bk)</span>"
      : "—";
    var vsOppCell = vsOpp.avg_stat != null
      ? fmt1(vsOpp.avg_stat) + ' <span class="enriched-dim">(' + (vsOpp.n_games || 0) + "g)</span>"
      : "—";

    html += "<tr>" +
      "<td>" + htmlBookPill(r.platform) + "</td>" +
      "<td>" + escapeHtml(r.market || "") + "</td>" +
      '<td class="num">' + fmt1(r.dfs_line) + "</td>" +
      '<td class="num">' + htmlEdgeCell(r) + "</td>" +
      '<td class="enriched-lean">' + modelCell + "</td>" +
      '<td class="enriched-lean">' + sharpCell + "</td>" +
      '<td class="col-secondary">' + consensusCell + "</td>" +
      '<td class="num col-secondary">' + fmtNumOrDash(model.stat_q50) + "</td>" +
      '<td class="num col-secondary">' + fmtNumOrDash(model.min_q50) + "</td>" +
      '<td class="num">' + fmtOverRatePct(form.over_l5) + "</td>" +
      '<td class="num">' + fmtOverRatePct(form.over_l10) + "</td>" +
      '<td class="num col-secondary">' + fmtOverRatePct(form.over_l15) + "</td>" +
      '<td class="num enriched-vsopp col-secondary">' + vsOppCell + "</td>" +
      '<td class="num col-secondary">' + fmtOrdinalRank(gc.opp_def_rating_rank) + "</td>" +
      "<td>" + htmlTierPill(r.tier) + "</td>" +
      "</tr>";
  }
  html += "</tbody></table></div>";

  var cards = '<div class="props-cards" aria-hidden="true">';
  for (var j = 0; j < sorted.length; j++) {
    var p = sorted[j];
    var m = p.model || {};
    var f = p.form || {};
    cards += '<div class="props-cards-prop">' +
      '<div class="props-cards-prop-header">' +
        '<span class="props-cards-prop-mkt">' + escapeHtml(p.market) + '</span>' +
        '<span class="props-cards-prop-line">' + fmt1(p.dfs_line) + '</span>' +
        htmlBestSidePill(p) +
        htmlBookPill(p.platform) +
        htmlTierPill(p.tier) +
      '</div>' +
      '<div class="props-cards-prop-meta">' +
        '<span>Edge ' + htmlEdgeCell(p) + '</span>' +
        '<span>Model ' + (m.lean || "—") + ' ' + fmtOverRatePct(m.p_over) + '</span>' +
        '<span>L5/10: ' + fmtOverRatePct(f.over_l5) + ' \xb7 ' + fmtOverRatePct(f.over_l10) + '</span>' +
      '</div>' +
    '</div>';
  }
  cards += '</div>';

  return html + cards;
}

function renderPlayersPanelGrouped() {
  var el = document.getElementById("playersPanel");
  if (!el) return;
  if (!ALL_ENRICHED.length) {
    el.innerHTML = '<p class="load-msg load-err">No player data found. ' +
      'Ensure <code>dfs_enriched_YYYYMMDD.json</code> exists under ' +
      '<code>data/props/enriched/</code>.</p>';
    return;
  }
  var searchEl = document.getElementById("playerSearch");
  var q = searchEl ? searchEl.value : "";

  var picks = ALL_ENRICHED.slice();
  picks = filterByPlatform(picks, activePlatform);
  picks = filterByTier(picks, activeTier);
  picks = filterPlayerRowsByStat(picks, activePlayerStat);
  picks = filterPlayerRows(picks, q);

  var players = sortPlayerGroups(aggregateEnrichedByPlayer(picks));

  if (!players.length) {
    el.innerHTML = '<div class="players-empty-state">No players match your filters.</div>';
    return;
  }

  var html =
    '<div class="players-grouped">' +
      '<div class="players-grouped-header">' +
        '<div class="players-grouped-count"><b>' + players.length + '</b> ' +
          (players.length === 1 ? "player" : "players") +
          ' \xb7 <span class="dim">' + picks.length + ' props</span></div>' +
        '<div class="players-sort-bar" role="group" aria-label="Sort players">' +
          '<span>Sort</span>' +
          sortGroupBtnHtml("edge_desc", "Best edge") +
          sortGroupBtnHtml("props_desc", "Most props") +
          sortGroupBtnHtml("verified_desc", "Most verified") +
          sortGroupBtnHtml("name_asc", "Name") +
        '</div>' +
      '</div>';

  for (var i = 0; i < players.length; i++) {
    var p = players[i];
    var isOpen = expandedPlayer === p.player;
    html += '<div class="player-block' + (isOpen ? " player-block--expanded" : "") +
            '" data-player="' + escapeHtml(p.player) + '">';
    html += htmlPlayerSummaryRow(p);
    if (isOpen) html += '<div class="player-block-body">' + htmlPlayerExpandedBody(p.picks) + '</div>';
    html += '</div>';
  }
  html += '</div>';
  el.innerHTML = html;
}

renderPlayersPanel = renderPlayersPanelGrouped;

function initPlayerGroupExpand() {
  var sec = document.getElementById("sectionPlayers");
  if (!sec) return;
  sec.addEventListener("click", function (e) {
    var row = e.target && e.target.closest(".player-row");
    if (!row || !sec.contains(row)) return;
    if (e.target.closest("[data-players-sort]") || e.target.closest("[data-grp-sort]")) return;
    var name = row.getAttribute("data-player");
    if (!name) return;
    expandedPlayer = (expandedPlayer === name) ? null : name;
    if (enrichedState === "loaded") renderPlayersPanel();
  });
}

function initPlayerGroupSort() {
  var sec = document.getElementById("sectionPlayers");
  if (!sec) return;
  sec.addEventListener("click", function (e) {
    var btn = e.target && e.target.closest("[data-grp-sort]");
    if (!btn || !sec.contains(btn)) return;
    e.stopPropagation();
    var k = btn.getAttribute("data-grp-sort");
    if (k && k !== playersGroupSort) {
      playersGroupSort = k;
      if (enrichedState === "loaded") renderPlayersPanel();
    }
  });
}

function initPlayerSearch() {
  var input = document.getElementById("playerSearch");
  if (!input) return;
  input.addEventListener("input", function () {
    if (enrichedState === "loaded") renderPlayersPanel();
  });
}

function initPlayersColumnSort() {
  var sec = document.getElementById("sectionPlayers");
  if (!sec) return;
  sec.addEventListener("click", function (e) {
    var btn = e.target && e.target.closest("[data-players-sort]");
    if (!btn || !sec.contains(btn)) return;
    var key = btn.getAttribute("data-players-sort");
    if (!key) return;
    if (playersSortKey === key) {
      playersSortDir = playersSortDir === "desc" ? "asc" : "desc";
    } else {
      playersSortKey = key;
      playersSortDir = "desc";
    }
    if (enrichedState === "loaded") renderPlayersPanel();
  });
}

function syncStatPillUi() {
  var pills = document.querySelectorAll("[data-stat]");
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
    var t = e.target && e.target.closest("[data-stat]");
    if (!t) return;
    var s = t.getAttribute("data-stat");
    if (s === "ALL") {
      activePlayerStat = null;
    } else {
      activePlayerStat = activePlayerStat === s ? null : s;
    }
    syncStatPillUi();
    if (enrichedState === "loaded") renderPlayersPanel();
  });
}

function syncPlatformPillUi() {
  var pills = document.querySelectorAll("[data-platform]");
  for (var i = 0; i < pills.length; i++) {
    var p = pills[i].getAttribute("data-platform");
    var on = p === "ALL" ? !activePlatform : p === activePlatform;
    pills[i].classList.toggle("active", on);
    pills[i].setAttribute("aria-pressed", on ? "true" : "false");
  }
}

function initPlatformFilter() {
  var bar = document.getElementById("platformFilter");
  if (!bar) return;
  bar.addEventListener("click", function (e) {
    var t = e.target && e.target.closest("[data-platform]");
    if (!t) return;
    var p = t.getAttribute("data-platform");
    if (p === "ALL") {
      activePlatform = null;
    } else {
      activePlatform = activePlatform === p ? null : p;
    }
    syncPlatformPillUi();
    if (enrichedState === "loaded") renderPlayersPanel();
  });
}

function syncTierPillUi() {
  var pills = document.querySelectorAll("[data-tier]");
  for (var i = 0; i < pills.length; i++) {
    var t = pills[i].getAttribute("data-tier");
    var on = t === "ALL" ? !activeTier : t === activeTier;
    pills[i].classList.toggle("active", on);
    pills[i].setAttribute("aria-pressed", on ? "true" : "false");
  }
}

function initTierFilter() {
  var bar = document.getElementById("tierFilter");
  if (!bar) return;
  bar.addEventListener("click", function (e) {
    var t = e.target && e.target.closest("[data-tier]");
    if (!t) return;
    var v = t.getAttribute("data-tier");
    if (v === "ALL") {
      activeTier = null;
    } else {
      activeTier = activeTier === v ? null : v;
    }
    syncTierPillUi();
    if (enrichedState === "loaded") renderPlayersPanel();
  });
}

function currentSlate() {
  var bucket = SLATES[activeLegs];
  if (!bucket) return [];
  return bucket[activeBook] || [];
}

function ordSuffix(n) {
  if (n == null || n === "" || isNaN(Number(n))) return "—";
  const i = Math.floor(Math.abs(Number(n)));
  const j = i % 100;
  if (j >= 11 && j <= 13) return i + "th";
  switch (i % 10) {
    case 1: return i + "st";
    case 2: return i + "nd";
    case 3: return i + "rd";
    default: return i + "th";
  }
}

function fmtOrdinalRank(n) {
  if (n == null || n === "") return "—";
  var v = Number(n);
  if (isNaN(v)) return "—";
  return ordSuffix(v);
}

function spreadFmt(n) {
  if (n == null || n === "" || isNaN(Number(n))) return "—";
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

/** One decimal, or em dash if missing / invalid. */
function fmtNumOrDash(x) {
  if (x == null || x === "") return "—";
  var n = Number(x);
  if (isNaN(n)) return "—";
  return fmt1(n);
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

/** Row label for last-10 stat average from slate MARKET (PTS / AST / REB / …). */
function l10StatAvgLabel(market) {
  var m = String(market || "")
    .trim()
    .toUpperCase();
  if (m === "PTS") return "L10 PTS Avg.";
  if (m === "AST") return "L10 AST Avg.";
  if (m === "REB" || m === "REBS") return "L10 REB Avg.";
  if (!m) return "L10 Stat Avg.";
  return "L10 " + escapeHtml(String(market).trim()) + " Avg.";
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
  const oppPaceRank = leg.oppPaceRank;
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
        "<span>" + l10StatAvgLabel(market) + "</span><span>" + fmtNumOrDash(avgL10) + "</span>" +
        "<span>vs matchup</span><span>" +
        fmtNumOrDash(avgVs) +
        (matchupGames != null &&
        String(matchupGames).trim() !== "" &&
        !isNaN(Number(matchupGames))
          ? " (" + matchupGames + " games)"
          : "") +
        "</span>" +
        "<span>Opp Pace Rank</span><span>" + fmtOrdinalRank(oppPaceRank) + "</span>" +
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
    paceRank1: row["OPP_PACE_RANK 1"],
    avgStatL101: row["AVG_STAT_L10 1"],
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
    paceRank2: row["OPP_PACE_RANK 2"],
    avgStatL102: row["AVG_STAT_L10 2"],
    avgVsMatchup2: row["AVG_STAT_VS_MATCHUP 2"],
    matchupGames2: row["MATCHUP_GAMES 2"],
    overRateL102: row["OVER_RATE_L10 2"],
    parlayProb: row["PARLAY_PROB"],
    ev: row["EV"],
    kelly: row["KELLY"]
  };
}

function legFromRow(row, i) {
  return {
    name: row["NAME " + i],
    team: row["TEAM " + i],
    market: row["MARKET " + i],
    line: row["LINE " + i],
    side: row["SIDE " + i],
    prediction: row["PREDICTION " + i],
    opponent: row["OPPONENT " + i],
    spread: row["SPREAD " + i],
    total: row["TOTAL " + i],
    defRank: row["OPP_DEF_RANK " + i],
    avgStatL10: row["AVG_STAT_L10 " + i],
    oppPaceRank: row["OPP_PACE_RANK " + i],
    avgVsMatchup: row["AVG_STAT_VS_MATCHUP " + i],
    matchupGames: row["MATCHUP_GAMES " + i],
    overRateL10: row["OVER_RATE_L10 " + i]
  };
}

function mapRowN(row, nLegs) {
  var legs = [];
  for (var i = 1; i <= nLegs; i++) {
    legs.push(legFromRow(row, i));
  }
  return {
    parlayProb: row["PARLAY_PROB"],
    ev: row["EV"],
    kelly: row["KELLY"],
    legs: legs
  };
}

function renderParlayCard(mapped, rank, nLegs) {
  var evNum = Number(mapped.ev);
  var evStr = (evNum >= 0 ? "+" : "-") + fmtEv(Math.abs(evNum));
  var probPct = pct0(mapped.parlayProb);
  var kellyPct = fmt1(mapped.kelly);
  var cardClass = nLegs > 2 ? "card card--" + nLegs + "leg" : "card";
  var footerClass = nLegs > 2 ? "card-footer card-footer--multi" : "card-footer";

  var legsHtml = "";
  var footerHtml = "";
  for (var i = 0; i < mapped.legs.length; i++) {
    legsHtml += renderLeg(mapped.legs[i]);
    var L = mapped.legs[i];
    footerHtml +=
      "<div>Game total " + fmtNumOrDash(L.total) + "<br />Spread " + spreadFmt(L.spread) + "</div>";
  }

  return (
    '<article class="' + cardClass + '">' +
      '<div class="card-header">' +
        '<span class="rank-label">#' + rank + " pick</span>" +
        '<div class="badges">' +
          '<span class="pill pill-ev">EV ' + evStr + "%</span>" +
          '<span class="pill pill-prob">Hit prob ' + probPct + "%</span>" +
          '<span class="pill pill-kelly">Kelly ' + kellyPct + "%</span>" +
        "</div>" +
      "</div>" +
      '<div class="legs">' + legsHtml + "</div>" +
      '<div class="' + footerClass + '">' + footerHtml + "</div>" +
    "</article>"
  );
}

function render() {
  if (activeView !== "pairs") return;
  const el = document.getElementById("cards");
  if (SLATE_LEG_COUNTS.indexOf(activeLegs) === -1) {
    var legLabel = legsDisplayLabel(activeLegs);
    el.innerHTML =
      '<p class="load-msg">No ' +
      escapeHtml(legLabel) +
      " slate is available yet. Export JSON for this leg count into <code>data/props/ev_analysis/</code>.</p>";
    return;
  }
  const sorted = currentSlate();
  if (!sorted.length) {
    var bookLabel = bookDisplayLabel(activeBook);
    el.innerHTML =
      '<p class="load-msg">No parlays in the ' + bookLabel + " slate. Export <code>" +
      activeSlateJsonFile() +
      "</code> into <code>data/props/ev_analysis/</code>.</p>";
    return;
  }
  let html = "";
  for (let i = 0; i < sorted.length; i++) {
    html += renderParlayCard(mapRowN(sorted[i], activeLegs), i + 1, activeLegs);
  }
  el.innerHTML = html;
}


initMainNav();
initPlayerSearch();
initPlayerStatFilter();
initPlatformFilter();
initTierFilter();
initPlayersColumnSort();
initPlayerGroupExpand();
initPlayerGroupSort();
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