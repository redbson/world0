// Progressive enhancement only: language switch, topic filters, archive filters.
// Without JS the page shows both languages and all rows.
(function () {
  var root = document.documentElement, KEY = "w0-lang";
  function setLang(l) {
    root.setAttribute("data-lang", l); root.lang = l === "zh" ? "zh" : "en";
    document.querySelectorAll(".langsw button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.l === l)); });
    try { localStorage.setItem(KEY, l); } catch (e) {}
  }
  var saved = "both"; try { saved = localStorage.getItem(KEY) || "both"; } catch (e) {}
  document.querySelectorAll(".langsw button").forEach(function (b) { b.addEventListener("click", function () { setLang(b.dataset.l); }); });
  setLang(["en", "zh", "both"].indexOf(saved) >= 0 ? saved : "both");

  // Topic filter bars: buttons are built from the data-cat values of the target elements.
  document.querySelectorAll(".filters[data-target]").forEach(function (bar) {
    var items = document.querySelectorAll(bar.dataset.target), cats = {};
    items.forEach(function (el) { (el.dataset.cat || "").split("|").forEach(function (c) { if (c) cats[c] = 1; }); });
    var names = Object.keys(cats).sort(); if (names.length < 2) return;
    function mk(label, val) {
      var b = document.createElement("button"); b.textContent = label; b.dataset.v = val; b.setAttribute("aria-pressed", String(val === ""));
      b.addEventListener("click", function () {
        bar.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", "false"); }); b.setAttribute("aria-pressed", "true");
        items.forEach(function (el) { el.classList.toggle("hidden", val !== "" && (el.dataset.cat || "").split("|").indexOf(val) < 0); });
      }); bar.appendChild(b);
    }
    mk("all / 全部", ""); names.forEach(function (n) { mk(n, n); });
  });

  // Archive filters: selects/inputs with data-filter=cat|dir|imp|date against #reports tbody tr
  var rows = document.querySelectorAll("#reports tbody tr"), ctls = document.querySelectorAll("[data-filter]");
  function apply() {
    var f = {}; ctls.forEach(function (c) { f[c.dataset.filter] = c.value.trim(); });
    rows.forEach(function (r) {
      var ok = true;
      if (f.cat && (r.dataset.cat || "").split("|").indexOf(f.cat) < 0) ok = false;
      if (f.dir && (r.dataset.dir || "").split("|").indexOf(f.dir) < 0) ok = false;
      if (f.imp && r.dataset.imp !== f.imp) ok = false;
      if (f.date && (r.dataset.date || "").indexOf(f.date) !== 0) ok = false;
      r.classList.toggle("hidden", !ok);
    });
  }
  ctls.forEach(function (c) { c.addEventListener("change", apply); c.addEventListener("input", apply); });
})();
