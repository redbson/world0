/* World0 research portal: progressive enhancement only.
   Every page is complete without this script; it adds radar and timeline
   filters and a live GitHub Pages status read from the public GitHub API. */
(function () {
  "use strict";

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function show(el, on) { if (on) { el.removeAttribute("hidden"); } else { el.setAttribute("hidden", ""); } }

  $$("[data-enhance]").forEach(function (el) { el.removeAttribute("hidden"); });

  // A link to a record inside a collapsed section (resolved hypotheses) opens it.
  var reveal = function () {
    var id = decodeURIComponent(location.hash.slice(1));
    var el = id && document.getElementById(id);
    for (var d = el; d; d = d.parentElement) { if (d.tagName === "DETAILS") { d.open = true; } }
    if (el) { el.scrollIntoView(); }
  };
  window.addEventListener("hashchange", reveal);
  if (location.hash) { reveal(); }

  // Research radar: search, category, scope, sort.
  var table = $("#radar-table");
  if (table) {
    var tbody = table.tBodies[0];
    var rows = $$("tr", tbody);
    var q = $("#radar-q"), cat = $("#radar-cat"), scope = $("#radar-scope"), sort = $("#radar-sort"), count = $("#radar-count");
    var keys = {
      seen: function (r) { return r.dataset.seen + r.dataset.rel; },
      rel: function (r) { return r.dataset.rel + r.dataset.seen; },
      pub: function (r) { return (r.dataset.pub + "-00-00").slice(0, 10) + r.dataset.rel; }
    };
    var apply = function () {
      var text = (q.value || "").trim().toLowerCase();
      var key = keys[sort.value] || keys.seen;
      rows.slice().sort(function (a, b) { return key(b) < key(a) ? -1 : key(b) > key(a) ? 1 : 0; })
        .forEach(function (r) { tbody.appendChild(r); });
      var shown = 0;
      rows.forEach(function (r) {
        var ok = (!text || r.dataset.text.indexOf(text) !== -1) &&
          (!cat.value || r.dataset.cat === cat.value) &&
          (scope.value !== "latest" || r.dataset.latest === "1") &&
          (scope.value !== "high" || r.classList.contains("is-high"));
        r.style.display = ok ? "" : "none";
        if (ok) { shown += 1; }
      });
      count.textContent = shown + " / " + rows.length + " papers";
    };
    [q, cat, scope, sort].forEach(function (el) { el.addEventListener("input", apply); el.addEventListener("change", apply); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement && !/input|select|textarea/i.test(document.activeElement.tagName)) {
        e.preventDefault(); q.focus();
      }
    });
    apply();
  }

  // Research timeline: date, category, direction, importance.
  var list = $("#timeline-list");
  if (list) {
    var items = $$(".tl", list);
    var month = $("#tl-month"), tcat = $("#tl-cat"), dir = $("#tl-dir"), imp = $("#tl-imp"), tcount = $("#tl-count");
    var filter = function () {
      var shown = 0;
      items.forEach(function (li) {
        var ok = (!month.value || li.dataset.month === month.value) &&
          (!tcat.value || li.dataset.cats.indexOf("|" + tcat.value + "|") !== -1) &&
          (!dir.value || li.dataset.dir === dir.value) &&
          (!imp.value || li.dataset.imp === imp.value);
        show(li, ok);
        if (ok) { shown += 1; }
      });
      tcount.textContent = shown + " / " + items.length + " sessions";
    };
    [month, tcat, dir, imp].forEach(function (el) { el.addEventListener("change", filter); });
    filter();
  }

  // Live GitHub Pages status: the latest "pages build and deployment" run for the publishing branch.
  var live = $("[data-live-deploy]");
  var repo = document.body.dataset.repo, branch = document.body.dataset.branch;
  if (live && repo && branch && window.fetch) {
    live.textContent = "checking…";
    var url = "https://api.github.com/repos/" + repo + "/actions/runs?per_page=5&branch=" + encodeURIComponent(branch);
    fetch(url, { headers: { Accept: "application/vnd.github+json" } })
      .then(function (r) { if (!r.ok) { throw new Error("HTTP " + r.status); } return r.json(); })
      .then(function (data) {
        var run = (data.workflow_runs || []).filter(function (w) { return /pages build and deployment/i.test(w.name || ""); })[0];
        live.textContent = "";
        if (!run) { live.textContent = "no Pages build found"; return; }
        var state = run.status === "completed" ? (run.conclusion || "unknown") : run.status;
        var chip = document.createElement("span");
        chip.className = "chip " + (state === "success" ? "st-COMPLETE" : state === "failure" ? "st-FAILED" : "st-PARTIAL");
        chip.textContent = state.toUpperCase();
        var when = document.createElement("time");
        when.dateTime = run.updated_at;
        when.textContent = " " + String(run.updated_at || "").slice(0, 16).replace("T", " ") + " UTC ";
        var sha = document.createElement("a");
        sha.className = "mono";
        sha.href = run.html_url;
        sha.rel = "noopener noreferrer external";
        sha.textContent = String(run.head_sha || "").slice(0, 7);
        live.appendChild(chip); live.appendChild(when); live.appendChild(sha);
      })
      .catch(function () { live.innerHTML = '<span class="dim">unavailable (GitHub API not reachable)</span>'; });
  }
})();
