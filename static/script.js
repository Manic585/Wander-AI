const form = document.querySelector("#travel-form");
const submitButton = document.querySelector("#submit-button");
const userInput = document.querySelector("#user-input");
const threadInput = document.querySelector("#thread-id");
const resultsPanel = document.querySelector("#results");
const toast = document.querySelector("#toast");
const approvalPanel = document.querySelector("#approval-panel");
const approvalQuestion = document.querySelector("#approval-question");
const approvalDraft = document.querySelector("#approval-draft");
const approvalForm = document.querySelector("#approval-form");
const approvalFeedback = document.querySelector("#approval-feedback");
const approveButton = document.querySelector("#approve-button");
const reviseButton = document.querySelector("#revise-button");

const fields = {
  answer: document.querySelector("#answer"),
  flights: document.querySelector("#flights"),
  hotels: document.querySelector("#hotels"),
  weather: document.querySelector("#weather"),
  budget: document.querySelector("#budget"),
  itinerary: document.querySelector("#itinerary"),
  thread: document.querySelector("#thread-label"),
  calls: document.querySelector("#calls-label"),
};

const steps = ["flight", "hotel", "weather", "budget", "itinerary", "final"];
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

function pauseForApproval() {
  window.clearInterval(progressTimer);
  ["flight", "hotel", "weather", "budget", "itinerary"].forEach((step) => {
    updateStep(step, "done", "Complete");
  });
  updateStep("final", "active", "Waiting for approval");
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

function markdownCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") {
    return JSON.stringify(value).replaceAll("|", "\\|");
  }
  return String(value).replaceAll("|", "\\|").replaceAll("\n", " ");
}

function structuredToMarkdown(value, title = "") {
  if (Array.isArray(value)) {
    if (!value.length) return title ? `### ${title}\n\nNo details returned.` : "No details returned.";

    if (value.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
      const columns = [...new Set(value.flatMap((item) => Object.keys(item)))];
      const header = `| ${columns.join(" | ")} |`;
      const divider = `| ${columns.map(() => "---").join(" | ")} |`;
      const rows = value.map(
        (item) => `| ${columns.map((column) => markdownCell(item[column])).join(" | ")} |`
      );
      return `${title ? `### ${title}\n\n` : ""}${header}\n${divider}\n${rows.join("\n")}`;
    }

    return `${title ? `### ${title}\n\n` : ""}${value.map((item) => `- ${markdownCell(item)}`).join("\n")}`;
  }

  if (value && typeof value === "object") {
    const sections = [];
    const facts = [];

    Object.entries(value).forEach(([key, item]) => {
      const label = key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
      if (item && typeof item === "object") {
        sections.push(structuredToMarkdown(item, label));
      } else {
        facts.push(`**${label}:** ${markdownCell(item)}`);
      }
    });

    return [title ? `### ${title}` : "", facts.join("\n\n"), ...sections]
      .filter(Boolean)
      .join("\n\n");
  }

  return markdownCell(value);
}

function parseJsonText(value) {
  const text = String(value ?? "").trim();
  if (!(text.startsWith("{") || text.startsWith("["))) return null;

  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function unwrapMcpTextBlocks(value) {
  if (Array.isArray(value)) {
    const textBlocks = value.filter(
      (item) => item && item.type === "text" && typeof item.text === "string"
    );

    if (textBlocks.length === value.length && textBlocks.length > 0) {
      const text = textBlocks.map((item) => item.text).join("\n").trim();
      return parseJsonText(text) ?? text;
    }
  }

  if (value && typeof value === "object" && value.type === "text" && typeof value.text === "string") {
    return parseJsonText(value.text) ?? value.text;
  }

  return value;
}

function responseToMarkdown(value) {
  const unwrapped = unwrapMcpTextBlocks(value);
  if (unwrapped !== value) return responseToMarkdown(unwrapped);
  if (value && typeof value === "object") return structuredToMarkdown(value);

  const text = String(value ?? "");
  const parsed = parseJsonText(text);
  if (parsed !== null) return responseToMarkdown(parsed);

  if (text.includes("Current Weather:") && text.includes("Forecast:")) {
    const currentStart = text.indexOf("Current Weather:") + "Current Weather:".length;
    const forecastStart = text.indexOf("Forecast:", currentStart);
    const current = parseJsonText(text.slice(currentStart, forecastStart));
    const forecast = parseJsonText(text.slice(forecastStart + "Forecast:".length));

    if (current !== null || forecast !== null) {
      return [
        current !== null
          ? `### Current Weather\n\n${responseToMarkdown(current)}`
          : "",
        forecast !== null
          ? `### Forecast\n\n${responseToMarkdown(forecast)}`
          : "",
      ].filter(Boolean).join("\n\n");
    }
  }

  return text;
}

function formatResponse(value) {
  const rawText = responseToMarkdown(value);
  const text = rawText
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
  if (data.thread_id) threadInput.value = data.thread_id;
  fields.answer.innerHTML = formatResponse(data.answer);
  fields.flights.innerHTML = formatResponse(data.flight_results);
  fields.hotels.innerHTML = formatResponse(data.hotel_results);
  fields.weather.innerHTML = formatResponse(data.weather_results);
  fields.budget.innerHTML = formatResponse(data.budget_results);
  fields.itinerary.innerHTML = formatResponse(data.itinerary);
  renderApproval(data);
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

function renderApproval(data) {
  const payload = data.interrupt_payload;
  const requiresApproval = data.requires_approval === true && payload;

  approvalPanel.hidden = !requiresApproval;
  if (!requiresApproval) return;

  approvalQuestion.textContent = payload.question || "Please review the draft itinerary.";
  approvalDraft.innerHTML = formatResponse(payload.draft_itinerary || data.itinerary);
  approvalFeedback.value = "";
  approveButton.disabled = false;
  reviseButton.disabled = false;
}

async function submitApproval(event) {
  event.preventDefault();

  const submitter = event.submitter;
  const approved = submitter?.dataset.approved === "true";
  const human_feedback = approvalFeedback.value.trim();

  if (!approved && !human_feedback) {
    showToast("Add feedback before requesting changes.");
    approvalFeedback.focus();
    return;
  }

  approveButton.disabled = true;
  reviseButton.disabled = true;
  submitter.textContent = approved ? "Approving..." : "Sending...";

  try {
    const response = await fetch("/api/travel/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: threadInput.value.trim(),
        approved,
        human_feedback,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to resume the travel plan.");
    }

    completeSteps();
    renderResults(payload);
  } catch (error) {
    approveButton.disabled = false;
    reviseButton.disabled = false;
    submitter.textContent = approved ? "Approve itinerary" : "Request changes";
    showToast(error.message);
  }
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

    if (payload.requires_approval) {
      pauseForApproval();
    } else {
      completeSteps();
    }
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
approvalForm.addEventListener("submit", submitApproval);
