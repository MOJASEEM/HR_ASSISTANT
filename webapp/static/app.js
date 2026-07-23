const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const messagesEl = document.getElementById("messages");
const emptyState = document.getElementById("emptyState");
const chatScroll = document.getElementById("chatScroll");
const traceLedger = document.getElementById("traceLedger");

function scrollToBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function addUserMessage(text) {
  const msg = document.createElement("div");
  msg.className = "msg msg-user";
  msg.innerHTML = `<div class="msg-bubble"></div>`;
  msg.querySelector(".msg-bubble").textContent = text;
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function addPendingBotMessage() {
  const msg = document.createElement("div");
  msg.className = "msg msg-bot";
  msg.innerHTML = `<div class="msg-bubble pending">Thinking…</div>`;
  messagesEl.appendChild(msg);
  scrollToBottom();
  return msg;
}

function typeText(el, text, speed = 14) {
  return new Promise((resolve) => {
    let i = 0;
    el.textContent = "";
    const cursor = document.createElement("span");
    cursor.className = "type-cursor";
    el.appendChild(cursor);

    function step() {
      if (i < text.length) {
        cursor.insertAdjacentText("beforebegin", text[i]);
        i++;
        const delay = /[.,!?]/.test(text[i - 1]) ? speed * 6 : speed;
        setTimeout(step, delay + Math.random() * 8);
        scrollToBottom();
      } else {
        cursor.remove();
        resolve();
      }
    }
    step();
  });
}

async function renderBotMessage(msgEl, answer, sources) {
  const bubble = msgEl.querySelector(".msg-bubble");
  bubble.classList.remove("pending");

  await typeText(bubble, answer);

  if (sources && sources.length > 0) {
    const details = document.createElement("details");
    details.className = "msg-sources";
    const list = sources.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
    details.innerHTML = `<summary>Sources used (${sources.length})</summary><ol>${list}</ol>`;
    msgEl.appendChild(details);
  }
  scrollToBottom();
}
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function resetTraceLedger() {
  traceLedger.hidden = false;
  document.querySelectorAll(".trace-ticket").forEach((el) => {
    el.classList.remove("is-good", "is-bad", "is-active");
    el.querySelector(".trace-ticket-value").textContent = "…";
  });
}

function setTicket(step, value, status) {
  const el = document.querySelector(`.trace-ticket[data-step="${step}"]`);
  if (!el) return;
  el.querySelector(".trace-ticket-value").textContent = value;
  el.classList.remove("is-good", "is-bad", "is-active");
  if (status) el.classList.add(status);
}

function updateTraceLedger(trace) {
  // Router
  if (trace.route) {
    setTicket("route", trace.route, "is-active");
  } else {
    setTicket("route", "—");
  }

  // Retrieval
  if (typeof trace.chunks_retrieved === "number") {
    setTicket("retrieve", trace.chunks_retrieved, "is-active");
  } else {
    setTicket("retrieve", "n/a");
  }

  // Grading
  if (typeof trace.chunks_kept === "number") {
    const good = trace.chunks_kept > 0;
    setTicket("grade", trace.chunks_kept, good ? "is-good" : "is-bad");
  } else {
    setTicket("grade", "n/a");
  }

  // Hallucination check
  if (typeof trace.grounded === "boolean") {
    setTicket("verify", trace.grounded ? "Grounded" : "Flagged", trace.grounded ? "is-good" : "is-bad");
  } else {
    setTicket("verify", "n/a");
  }
}

async function sendQuestion(question) {
  emptyState.style.display = "none";
  addUserMessage(question);
  resetTraceLedger();

  const pendingMsg = addPendingBotMessage();
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      throw new Error(`Server responded with ${res.status}`);
    }

    const data = await res.json();
    updateTraceLedger(data.trace || {});
    await renderBotMessage(pendingMsg, data.answer, data.sources);
  } catch (err) {
    const bubble = pendingMsg.querySelector(".msg-bubble");
    bubble.classList.remove("pending");
    bubble.textContent = "Something went wrong reaching the assistant. Check that the server is running.";
  } finally {
    sendBtn.disabled = false;
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  sendQuestion(question);
});

document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const q = chip.getAttribute("data-q");
    sendQuestion(q);
  });
});
