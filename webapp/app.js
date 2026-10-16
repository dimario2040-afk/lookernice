/* MOG mini app — логика клиента */
(function () {
  "use strict";

  const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;
  if (tg) { try { tg.ready(); tg.expand(); } catch (e) {} }

  const $ = (id) => document.getElementById(id);
  const initData = tg && tg.initData ? tg.initData : "";
  const startParam = tg && tg.initDataUnsafe && tg.initDataUnsafe.start_param
    ? tg.initDataUnsafe.start_param : "";

  let state = {
    photoToken: null,
    battleId: null,
    rollsLeft: 3,
    bonusRolls: 0,
    critTokens: 0,
    catalog: {},
    lastRoll: null,
  };

  const STAT_NAMES = { jaw: "Челюсть", eyes: "Глаза", cheeks: "Скулы", harmony: "Гармония", groom: "Груминг", aura: "Аура" };

  // ---- api -----------------------------------------------------------------
  async function api(path, opts) {
    opts = opts || {};
    const headers = Object.assign(
      { "Content-Type": "application/json" },
      opts.headers || {},
      initData ? { "X-Init-Data": initData } : {}
    );
    const res = await fetch(path, Object.assign({}, opts, { headers }));
    let data = null;
    try { data = await res.json(); } catch (e) { data = {}; }
    return { ok: res.ok, status: res.status, data: data };
  }

  function haptic(type) {
    if (!tg || !tg.HapticFeedback) return;
    try {
      if (type === "ok") tg.HapticFeedback.notificationOccurred("success");
      else if (type === "err") tg.HapticFeedback.notificationOccurred("error");
      else tg.HapticFeedback.impactOccurred(type || "light");
    } catch (e) {}
  }

  function toast(msg, ms) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.remove("hidden");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.add("hidden"), ms || 2600);
  }

  function show(id) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
    $(id).classList.add("active");
  }

  // ---- me ------------------------------------------------------------------
  async function loadMe() {
    const r = await api("/api/me");
    if (r.status === 401) { toast("Открой через кнопку в Telegram-боте 🙂", 4000); return; }
    if (!r.ok) { toast("Ошибка загрузки профиля"); return; }
    const d = r.data;
    state.rollsLeft = d.rolls_left;
    state.bonusRolls = d.bonus_rolls || 0;
    state.critTokens = d.crit_tokens || 0;
    state.catalog = d.catalog || {};
    $("chip-rolls").textContent = d.rolls_left + state.bonusRolls;
    $("chip-crits").textContent = d.crits;
    $("chip-best").textContent = d.best_roll ? d.best_roll.score.toFixed(4) : "—";
    updateAgainPill();
    updateCritBtn();
  }

  function totalRolls() { return state.rollsLeft + state.bonusRolls; }

  function updateAgainPill() {
    const p = $("again-left");
    if (p) p.textContent = totalRolls();
  }

  function updateCritBtn() {
    const b = $("btn-crit");
    if (!b) return;
    b.classList.remove("hidden");
    b.textContent = state.critTokens > 0
      ? "💎 Крит-токен ×" + state.critTokens + " — применить"
      : "⭐ Купить крит-токен";
  }

  // ---- photo + roll --------------------------------------------------------
  $("btn-roll").addEventListener("click", () => { haptic("light"); pickPhoto(false); });
  $("btn-again").addEventListener("click", () => {
    haptic("light");
    if (totalRolls() <= 0) { toast("Роллы кончились. Пригласи друга или купи в Stars ⭐"); openShop(); return; }
    pickPhoto(false);
  });
  $("btn-battle-photo").addEventListener("click", () => { haptic("light"); pickPhoto(false); });

  $("btn-crit").addEventListener("click", () => {
    haptic("light");
    if (state.critTokens > 0) {
      if (!state.photoToken) { toast("Сначала кинь фото, потом применим крит"); pickPhoto(false); return; }
      doRoll(true);
    } else {
      toast("Нет крит-токенов — купи в магазине ⭐");
      openShop();
    }
  });

  function pickPhoto(useCrit) {
    $("file-input").dataset.useCrit = useCrit ? "1" : "";
    $("file-input").value = "";
    $("file-input").click();
  }

  $("file-input").addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    await handlePhoto(file);
  });

  function loadingText() {
    const lines = ["Могнём…", "Считаю кости…", "Ищу halo…", "Позирую в паке…", "Мог-машина крутится…"];
    return lines[Math.floor(Math.random() * lines.length)];
  }

  async function handlePhoto(file) {
    if (!initData) { toast("Открой через Telegram-бота, чтобы крутить 🎰", 4000); return; }
    $("loading-text").textContent = loadingText();
    show("screen-loading");
    haptic("medium");

    const b64 = await fileToB64(file);
    const up = await api("/api/photo", { method: "POST", body: JSON.stringify({ photo_b64: b64 }) });
    if (!up.ok) { show("screen-home"); toast("Не смог принять фото"); return; }
    state.photoToken = up.data.photo_token;

    const useCrit = $("file-input").dataset.useCrit === "1";
    $("file-input").dataset.useCrit = "";

    if (state.battleId) { await battleSubmit(); return; }
    await doRoll(useCrit);
  }

  function fileToB64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  async function doRoll(useCrit) {
    const r = await api("/api/roll", {
      method: "POST",
      body: JSON.stringify({ photo_token: state.photoToken, use_crit: !!useCrit })
    });
    if (r.status === 402) { show("screen-home"); toast("Лимит роллов. Пригласи друга или купи ⭐"); haptic("err"); return; }
    if (!r.ok) { show("screen-home"); toast("Ролл не прошёл, попробуй ещё"); return; }
    state.rollsLeft = r.data.rolls_left;
    state.bonusRolls = r.data.bonus_rolls || 0;
    state.critTokens = r.data.crit_tokens || 0;
    state.lastRoll = r.data;
    await animateRoll(r.data);
    renderCard(r.data);
    show("screen-card");
    haptic(r.data.crit ? "ok" : "light");
    if (r.data.crit_token_used) toast("💎 Крит-токен применён!");
    updateCritBtn();
    loadMe();
  }

  // слот-машина; тир/крит подсвечиваются ПОСЛЕ ответа сервера (анимация не врёт)
  function animateRoll(d) {
    return new Promise((resolve) => {
      const scoreEl = $("card-score");
      const start = 0.0, target = d.score;
      const t0 = performance.now(), dur = 750;
      function frame(t) {
        const k = Math.min(1, (t - t0) / dur);
        const eased = 1 - Math.pow(1 - k, 3);
        scoreEl.textContent = (start + (target - start) * eased).toFixed(4);
        if (k < 1) requestAnimationFrame(frame);
        else { scoreEl.textContent = target.toFixed(4); resolve(); }
      }
      requestAnimationFrame(frame);
    });
  }

  function renderCard(d) {
    const card = $("card");
    card.className = "card t-" + d.tier;
    $("card-tier").textContent = d.tier;
    $("card-score").textContent = d.score.toFixed(4);
    $("card-roast").textContent = d.roast || "";
    $("card-crit").classList.toggle("hidden", !d.crit);
    $("card-face").textContent = d.crit ? "💎" : (d.failed ? "💀" : "🫥");
    updateAgainPill();

    const bars = $("stat-bars");
    bars.innerHTML = "";
    Object.keys(STAT_NAMES).forEach((k) => {
      const v = d.stats && d.stats[k] != null ? d.stats[k] : 0;
      const div = document.createElement("div");
      div.className = "stat";
      div.innerHTML = STAT_NAMES[k] + " · " + v +
        '<div class="bar"><div class="fill" style="width:' + v + '%"></div></div>';
      bars.appendChild(div);
    });
  }

  // ---- share ---------------------------------------------------------------
  $("btn-share").addEventListener("click", async () => {
    haptic("light");
    const d = state.lastRoll;
    const text = d
      ? "Я могнул на " + d.score.toFixed(4) + " (" + d.tier + ")! А ты? 🎰"
      : "Проверь себя в MOG 🎰";
    await shareText(text);
  });

  async function shareText(text) {
    const url = "https://t.me/share/url?url=" + encodeURIComponent("https://t.me/lookernice_bot") + "&text=" + encodeURIComponent(text);
    if (navigator.share) { try { await navigator.share({ text: text }); return; } catch (e) {} }
    if (tg && tg.openTelegramLink) tg.openTelegramLink(url);
    else window.open(url, "_blank");
  }

  // ---- shop (Telegram Stars) ----------------------------------------------
  $("btn-shop").addEventListener("click", () => { haptic("light"); openShop(); });
  $("btn-home3").addEventListener("click", () => show("screen-home"));

  function openShop() {
    const list = $("shop-list");
    list.innerHTML = "";
    const keys = Object.keys(state.catalog || {});
    if (!keys.length) { list.innerHTML = "<li>Магазин пуст.</li>"; }
    keys.forEach((key) => {
      const p = state.catalog[key];
      const li = document.createElement("li");
      li.innerHTML = '<span class="nm">' + esc(p.title) + '</span>' +
        '<span class="sc">' + p.stars + ' ⭐</span>';
      li.style.cursor = "pointer";
      li.addEventListener("click", () => buy(key, p));
      list.appendChild(li);
    });
    show("screen-shop");
  }

  async function buy(key, product) {
    haptic("light");
    if (!tg || !tg.openInvoice) { toast("Оплата доступна только внутри Telegram"); return; }
    const r = await api("/api/buy", { method: "POST", body: JSON.stringify({ product: key }) });
    if (!r.ok || !r.data.invoice_link) { toast("Не смог выставить счёт"); haptic("err"); return; }
    tg.openInvoice(r.data.invoice_link, (status) => {
      if (status === "paid") { haptic("ok"); toast("Оплачено! " + product.title); loadMe(); }
      else if (status === "failed") { haptic("err"); toast("Оплата не прошла"); }
    });
  }

  // ---- battle --------------------------------------------------------------
  $("btn-check").addEventListener("click", () => { haptic("light"); shareText("Проверь себя в MOG — я уже могнул. Кидай фото: @lookernice_bot"); });

  async function maybeBattle() {
    if (!startParam || startParam.indexOf("battle_") !== 0) return false;
    state.battleId = parseInt(startParam.replace("battle_", ""), 10);
    if (!state.battleId) return false;
    const r = await api("/api/battle/" + state.battleId);
    if (!r.ok) return false;
    const b = r.data;
    $("battle-info").innerHTML =
      "Ты: <b>" + esc(b.opponent.name) + "</b><br>Против: <b>" + esc(b.challenger.name) +
      "</b> (" + Number(b.challenger.score).toFixed(4) + ")<br><br>" +
      "Кидай своё фото — кто кого смог.";
    if (b.status !== "pending") {
      $("battle-info").innerHTML += "<br><br>Этот батл уже закрыт.";
      $("btn-battle-photo").classList.add("hidden");
    }
    show("screen-battle");
    return true;
  }

  async function battleSubmit() {
    const r = await api("/api/battle/" + state.battleId + "/photo", {
      method: "POST", body: JSON.stringify({ photo_token: state.photoToken })
    });
    if (r.status === 402) { show("screen-home"); state.battleId = null; toast("Нет роллов — пригласи друга или купи ⭐"); return; }
    if (!r.ok) { show("screen-home"); state.battleId = null; toast("Батл не прошёл"); return; }
    const res = r.data.result;
    const win = r.data.winner_id;
    const myId = tg && tg.initDataUnsafe && tg.initDataUnsafe.user ? tg.initDataUnsafe.user.id : null;
    $("battle-info").innerHTML =
      "Ты: <b>" + Number(res.opponent_score).toFixed(4) + "</b><br>" +
      "Против: <b>" + Number(res.challenger_score).toFixed(4) + "</b><br><br>" +
      (win === null ? "🤝 Ничья!" : (win === myId ? "💀 ТЫ СМОГНУЛ!" : "💀 Тебя смогнули"));
    $("btn-battle-photo").classList.add("hidden");
    state.battleId = null;
    show("screen-battle");
    haptic("ok");
  }

  // ---- leaderboard ---------------------------------------------------------
  $("btn-board").addEventListener("click", async () => {
    haptic("light");
    const r = await api("/api/leaderboard");
    const list = $("board-list");
    list.innerHTML = "";
    if (r.ok && r.data.top) {
      r.data.top.forEach((row) => {
        const li = document.createElement("li");
        li.innerHTML = '<span class="rk">' + row.rank + '</span><span class="nm">' + esc(row.name) +
          '</span><span class="sc">' + Number(row.score).toFixed(4) + '</span>';
        list.appendChild(li);
      });
      if (!r.data.top.length) list.innerHTML = "<li>Пока пусто — будь первым могером.</li>";
    } else { list.innerHTML = "<li>Недоступно.</li>"; }
    show("screen-board");
  });

  $("btn-home").addEventListener("click", () => show("screen-home"));
  $("btn-home2").addEventListener("click", () => show("screen-home"));

  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  // ---- boot ----------------------------------------------------------------
  (async function boot() {
    if (tg && tg.themeParams && tg.themeParams.bg_color) {
      document.documentElement.style.setProperty("--bg", tg.themeParams.bg_color);
    }
    const inBattle = await maybeBattle();
    if (!inBattle) { await loadMe(); show("screen-home"); }
  })();
})();
