// שלוש – minimal vanilla JS: interview streaming, countdowns, copy-to-clipboard.
(function () {
  "use strict";
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";

  // ------------------------------------------------------------- countdown
  document.querySelectorAll("[data-countdown]").forEach((el) => {
    const target = new Date(el.dataset.countdown).getTime();
    const d = el.querySelector("[data-d]"), h = el.querySelector("[data-h]"), m = el.querySelector("[data-m]");
    const tick = () => {
      let s = Math.max(0, Math.floor((target - Date.now()) / 1000));
      const days = Math.floor(s / 86400); s -= days * 86400;
      const hours = Math.floor(s / 3600); s -= hours * 3600;
      const mins = Math.floor(s / 60);
      if (d) d.textContent = String(days);
      if (h) h.textContent = String(hours).padStart(2, "0");
      if (m) m.textContent = String(mins).padStart(2, "0");
    };
    tick();
    setInterval(tick, 20000);
  });
  document.querySelectorAll("[data-hours-left]").forEach((el) => {
    const target = new Date(el.dataset.hoursLeft).getTime();
    const hrs = Math.max(0, Math.floor((target - Date.now()) / 3600000));
    el.textContent = String(hrs);
  });

  // ------------------------------------------------------------- copy buttons
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(btn.dataset.copy);
        const old = btn.textContent; btn.textContent = "הועתק ✓";
        setTimeout(() => (btn.textContent = old), 1500);
      } catch (e) { window.prompt("העתיקו את הקישור:", btn.dataset.copy); }
    });
  });

  // forms that take a while (Claude extraction, uploads): show progress, avoid double submit
  document.querySelectorAll("form[data-busy]").forEach((f) =>
    f.addEventListener("submit", () => {
      const b = f.querySelector("button[type=submit], button:not([type])");
      if (b) setTimeout(() => { b.disabled = true; b.textContent = f.dataset.busy; }, 0);
    }));

  // ------------------------------------------------------------- interview
  const chat = document.getElementById("chat");
  if (!chat) return;
  const form = document.getElementById("composer");
  const input = document.getElementById("msg-input");
  const send = document.getElementById("send-btn");
  const finishBox = document.getElementById("finish-box");
  const earlyFinish = document.getElementById("early-finish");
  const turnLabel = document.getElementById("turn-label");
  const bar = document.getElementById("progress-bar");
  const maxTurns = parseInt(chat.dataset.maxTurns, 10);
  const minTurns = parseInt(chat.dataset.minTurns, 10);
  let turns = parseInt(chat.dataset.turns, 10);
  let busy = false;

  const scroll = () => { chat.scrollTop = chat.scrollHeight; };
  const progress = () => {
    if (bar) bar.style.width = Math.min(100, Math.round((turns / maxTurns) * 100)) + "%";
    if (turnLabel) turnLabel.textContent = turns + "/" + maxTurns;
    if (earlyFinish) earlyFinish.hidden = turns < minTurns;
  };
  const bubble = (role, text) => {
    const div = document.createElement("div");
    div.className = "msg " + role;
    div.textContent = text;
    chat.appendChild(div);
    scroll();
    return div;
  };
  const ended = () => {
    form.hidden = true;
    finishBox.hidden = false;
    if (earlyFinish) earlyFinish.hidden = true;
  };

  async function ask(text) {
    busy = true; send.disabled = true; input.disabled = true;
    if (text) { bubble("user", text); turns += 1; progress(); }
    const out = bubble("assistant", "");
    out.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
    let got = "";
    try {
      const resp = await fetch("/p/interview/message", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({ text: text || "" }),
      });
      if (!resp.ok || !resp.body) throw new Error("HTTP " + resp.status);
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, idx).trim(); buf = buf.slice(idx + 2);
          if (!line.startsWith("data:")) continue;
          const ev = JSON.parse(line.slice(5));
          if (ev.delta) { got += ev.delta; out.textContent = got; scroll(); }
          if (ev.error) { out.textContent = ev.error; out.dataset.retry = "1"; }
          if (ev.done && ev.ended) ended();
        }
      }
    } catch (e) {
      out.textContent = "החיבור נקטע. לחצו על שליחה כדי לנסות שוב.";
      out.dataset.retry = "1";
    }
    busy = false; send.disabled = false; input.disabled = false; input.focus();
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (busy) return;
    const retry = chat.querySelector(".msg.assistant:last-child[data-retry]");
    const text = input.value.trim();
    if (retry) { retry.remove(); ask(""); return; }   // server re-generates the missing reply
    if (!text) return;
    input.value = "";
    if (turns >= maxTurns) return;
    ask(text);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  progress();
  scroll();
  if (chat.dataset.ended === "1") ended();
  else if (chat.dataset.needReply === "1") ask("");
})();
