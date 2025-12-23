const API_URL = "http://localhost:8000";

let currentMode = "statement";

const contentInput = document.getElementById("content-input");
const inputLabel = document.getElementById("input-label");
const analyzeBtn = document.getElementById("analyze-btn");
const resultsSection = document.getElementById("results");
const errorSection = document.getElementById("error");
const inputSection = document.querySelector(".input-section");

const modeStatementBtn = document.getElementById("mode-statement");
const modeArticleBtn = document.getElementById("mode-article");

const iconFake = document.getElementById("icon-fake");
const iconReal = document.getElementById("icon-real");
const verdictText = document.getElementById("verdict-text");
const confidenceValue = document.getElementById("confidence-value");
const progressCircle = document.getElementById("progress-circle");
const fakeBar = document.getElementById("fake-bar");
const realBar = document.getElementById("real-bar");
const fakePercent = document.getElementById("fake-percent");
const realPercent = document.getElementById("real-percent");
const previewLabel = document.getElementById("preview-label");
const analyzedContent = document.getElementById("analyzed-content");
const resetBtn = document.getElementById("reset-btn");
const errorMessage = document.getElementById("error-message");
const errorDismiss = document.getElementById("error-dismiss");

const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * 52;

analyzeBtn.addEventListener("click", handleAnalyze);
resetBtn.addEventListener("click", handleReset);
errorDismiss.addEventListener("click", handleReset);

modeStatementBtn.addEventListener("click", () => switchMode("statement"));
modeArticleBtn.addEventListener("click", () => switchMode("article"));

contentInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && currentMode === "statement") {
    e.preventDefault();
    handleAnalyze();
  }
});

function switchMode(mode) {
  currentMode = mode;

  modeStatementBtn.classList.toggle("active", mode === "statement");
  modeArticleBtn.classList.toggle("active", mode === "article");

  if (mode === "statement") {
    inputLabel.textContent = "Enter a statement to verify";
    contentInput.placeholder = "Paste or type a news statement here...";
    contentInput.rows = 4;
    analyzeBtn.querySelector(".btn-text").textContent = "Analyze Statement";
  } else {
    inputLabel.textContent = "Enter an article to verify";
    contentInput.placeholder = "Paste or type a news article here...";
    contentInput.rows = 8;
    analyzeBtn.querySelector(".btn-text").textContent = "Analyze Article";
  }

  contentInput.focus();
}

async function handleAnalyze() {
  const content = contentInput.value.trim();
  const isArticle = currentMode === "article";

  if (!content) {
    showError(
      `Please enter ${isArticle ? "an article" : "a statement"} to analyze.`
    );
    return;
  }

  setLoading(true);
  hideError();

  try {
    const endpoint = isArticle ? "/predict/article" : "/predict";
    const body = isArticle ? { article: content } : { statement: content };

    const response = await fetch(`${API_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    showResults(data, isArticle);
  } catch (error) {
    console.error("Analysis error:", error);

    if (error.message.includes("Failed to fetch")) {
      showError(
        "Unable to connect to the server. Please make sure the API is running."
      );
    } else {
      showError(
        error.message || "An unexpected error occurred. Please try again."
      );
    }
  } finally {
    setLoading(false);
  }
}

function showResults(data, isArticle = false) {
  const { prediction, confidence, probabilities } = data;
  const content = isArticle ? data.article : data.statement;
  const isFake = prediction === "Fake";

  inputSection.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  iconFake.classList.toggle("hidden", !isFake);
  iconReal.classList.toggle("hidden", isFake);

  verdictText.textContent = isFake ? "Likely Fake" : "Likely Real";
  verdictText.className = `verdict-text ${isFake ? "fake" : "real"}`;

  previewLabel.textContent = isArticle
    ? "Analyzed Article"
    : "Analyzed Statement";
  analyzedContent.textContent =
    content.length > 300 ? content.substring(0, 300) + "..." : content;

  const confidencePercent = Math.round(confidence * 100);
  progressCircle.className = `progress-ring-circle ${isFake ? "fake" : "real"}`;

  setTimeout(() => {
    const offset = CIRCLE_CIRCUMFERENCE - confidence * CIRCLE_CIRCUMFERENCE;
    progressCircle.style.strokeDashoffset = offset;

    animateValue(confidenceValue, 0, confidencePercent, 1200);

    const fakePercentValue = Math.round(probabilities.fake * 100);
    const realPercentValue = Math.round(probabilities.real * 100);

    fakeBar.style.width = `${fakePercentValue}%`;
    realBar.style.width = `${realPercentValue}%`;

    animateValue(fakePercent, 0, fakePercentValue, 1000, "%");
    animateValue(realPercent, 0, realPercentValue, 1000, "%");
  }, 100);
}

function animateValue(element, start, end, duration, suffix = "") {
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    const easeOut = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (end - start) * easeOut);

    element.textContent = current + suffix;

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

function handleReset() {
  progressCircle.style.strokeDashoffset = CIRCLE_CIRCUMFERENCE;
  fakeBar.style.width = "0";
  realBar.style.width = "0";
  confidenceValue.textContent = "0";
  fakePercent.textContent = "0%";
  realPercent.textContent = "0%";

  iconFake.classList.add("hidden");
  iconReal.classList.add("hidden");

  inputSection.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  errorSection.classList.add("hidden");

  contentInput.value = "";
  contentInput.focus();
}

function showError(message) {
  errorMessage.textContent = message;
  errorSection.classList.remove("hidden");
  resultsSection.classList.add("hidden");
}

function hideError() {
  errorSection.classList.add("hidden");
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  analyzeBtn.classList.toggle("loading", isLoading);
}

async function checkApiHealth() {
  try {
    const response = await fetch(`${API_URL}/`);
    const data = await response.json();

    if (!data.model_loaded) {
      console.warn("API is running but model is not loaded");
    }
  } catch (error) {
    console.warn("API health check failed:", error.message);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  checkApiHealth();
  contentInput.focus();
});
