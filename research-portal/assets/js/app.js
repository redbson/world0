// Language switch: EN / 中文 / both. Page is fully readable without JS (defaults to both).
(function () {
  var root = document.documentElement, KEY = "w0-lang";
  function set(l) {
    root.setAttribute("data-lang", l);
    root.lang = l === "zh" ? "zh" : "en";
    document.querySelectorAll(".langsw button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.l === l));
    });
    try { localStorage.setItem(KEY, l); } catch (e) {}
  }
  var saved = "both";
  try { saved = localStorage.getItem(KEY) || "both"; } catch (e) {}
  document.querySelectorAll(".langsw button").forEach(function (b) {
    b.addEventListener("click", function () { set(b.dataset.l); });
  });
  set(["en", "zh", "both"].indexOf(saved) >= 0 ? saved : "both");
})();
