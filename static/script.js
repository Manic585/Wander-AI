const form = document.querySelector("#travel-form");
const submitButton = document.querySelector("#submit-button");
const userInput = document.querySelector("#user-input");
const threadInput = document.querySelector("#thread-id");
const resultsPanel = document.querySelector("#results");
const toast = document.querySelector("#toast");

const fields = {
  answer: document.querySelector("#answer"),
  flights: document.querySelector("#flights"),
  hotels: document.querySelector("#hotels"),
  itinerary: document.querySelector("#itinerary"),
  thread: document.querySelector("#thread-label"),
  calls: document.querySelector("#calls-label"),
};

const steps = ["flight", "hotel", "itinerary", "final"];
let progressTimer = null;

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.hidden = true;
  }, 5200);
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.innerHTML = isLoading
    ? '<span class="button-icon" aria-hidden="true">✦</span>Planning...'
    : '<span class="button-icon" aria-hidden="true">✦</span>Generate Plan';
}

function updateStep(stepName, state, label) {
  const card = document.querySelector(`[data-step="${stepName}"]`);
  if (!card) return;

  card.classList.toggle("active", state === "active");
  card.classList.toggle("done", state === "done");
  card.querySelector("p").textContent = label;
}

function resetSteps() {
  steps.forEach((step) => updateStep(step, "idle", "Waiting"));
}

function animateProgress() {
  let index = 0;
  resetSteps();
  updateStep(steps[0], "active", "Running");

  progressTimer = window.setInterval(() => {
    updateStep(steps[index], "done", "Complete");
    index += 1;

    if (index >= steps.length) {
      window.clearInterval(progressTimer);
      return;
    }

    updateStep(steps[index], "active", "Running");
  }, 1600);
}

function completeSteps() {
  window.clearInterval(progressTimer);
  steps.forEach((step) => updateStep(step, "done", "Complete"));
}

function failActiveStep() {
  window.clearInterval(progressTimer);
  const active = document.querySelector(".step-card.active");
  if (active) {
    active.classList.remove("active");
    active.querySelector("p").textContent = "Needs attention";
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatResponse(value) {
  const text = String(value ?? "")
    .replace(/&lt;(\/?think\b.*?)&gt;/gi, "<$1>")
    .replace(/<think\b[^>]*>[\s\S]*?(?:<\/think\s*>|$)/gi, "")
    .replace(/<\/think\s*>/gi, "")
    .trim();
  if (!text) return "<p>No travel details returned. Please generate the plan again.</p>";

  if (!window.marked || !window.DOMPurify) {
    return `<p>${escapeHtml(text).replaceAll("\n", "<br>")}</p>`;
  }

  return DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: true }), {
    ALLOWED_TAGS: [
      "h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "hr",
      "strong", "em", "del", "ul", "ol", "li", "blockquote", "a",
      "table", "thead", "tbody", "tr", "th", "td", "pre", "code",
    ],
    ALLOWED_ATTR: ["href", "title", "start", "colspan", "rowspan"],
    ALLOW_DATA_ATTR: false,
    ALLOW_ARIA_ATTR: false,
  });
}

function renderResults(data) {
  fields.thread.textContent = `Thread: ${data.thread_id || "--"}`;
  fields.calls.textContent = `Calls: ${data.llm_calls ?? "--"}`;
  fields.answer.innerHTML = formatResponse(data.answer);
  fields.flights.innerHTML = formatResponse(data.flight_results);
  fields.hotels.innerHTML = formatResponse(data.hotel_results);
  fields.itinerary.innerHTML = formatResponse(data.itinerary);
  document.querySelectorAll(".result-content table").forEach((table) => {
    const wrapper = document.createElement("div");
    wrapper.className = "result-table";
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", "Travel details table");
    table.replaceWith(wrapper);
    wrapper.append(table);
  });
  resultsPanel.hidden = false;
  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function submitTravelRequest(event) {
  event.preventDefault();

  const user_input = userInput.value.trim();
  const thread_id = threadInput.value.trim();

  if (!user_input) {
    showToast("Enter a trip request first.");
    return;
  }

  setLoading(true);
  animateProgress();

  try {
    const response = await fetch("/api/travel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_input, thread_id: thread_id || null }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Unable to generate the travel plan.");
    }

    completeSteps();
    renderResults(payload);
  } catch (error) {
    failActiveStep();
    showToast(error.message);
  } finally {
    setLoading(false);
  }
}

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    userInput.value = button.dataset.prompt;
    userInput.focus();
  });
});

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab-button").forEach((tab) => {
      tab.classList.toggle("active", tab === button);
      tab.setAttribute("aria-selected", String(tab === button));
    });

    document.querySelectorAll(".result-content").forEach((panel) => {
      panel.classList.toggle("active", panel.id === button.dataset.target);
    });
  });
});

form.addEventListener("submit", submitTravelRequest);
