const API_URL = "http://localhost:8000";

const statementInput = document.getElementById("statement");
const analyzeBtn = document.getElementById("analyze-btn");
const resultsSection = document.getElementById("results");
const errorSection = document.getElementById("error");
const inputSection = document.querySelector(".input-section");

const verdictIcon = document.getElementById("verdict-icon");
const verdictText = document.getElementById("verdict-text");
const confidenceValue = document.getElementById("confidence-value");
const progressCircle = document.getElementById("progress-circle");
const fakeBar = document.getElementById("fake-bar");
const realBar = document.getElementById("real-bar");
const fakePercent = document.getElementById("fake-percent");
const realPercent = document.getElementById("real-percent");
const analyzedStatement = document.getElementById("analyzed-statement");
const resetBtn = document.getElementById("reset-btn");
const errorMessage = document.getElementById("error-message");
const errorDismiss = document.getElementById("error-dismiss");

const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * 52;

analyzeBtn.addEventListener("click", handleAnalyze);
resetBtn.addEventListener("click", handleReset);
errorDismiss.addEventListener("click", handleReset);

statementInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleAnalyze();
  }
});

async function handleAnalyze() {
  const statement = statementInput.value.trim();

  if (!statement) {
    showError("Please enter a statement to analyze.");
    return;
  }

  setLoading(true);
  hideError();

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ statement }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    showResults(data);
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

function showResults(data) {
  const { prediction, confidence, probabilities, statement } = data;
  const isFake = prediction === "Fake";

  inputSection.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  verdictIcon.textContent = isFake ? "🚫" : "✅";
  verdictText.textContent = isFake ? "Likely Fake" : "Likely Real";
  verdictText.className = `verdict-text ${isFake ? "fake" : "real"}`;

  analyzedStatement.textContent =
    statement.length > 300 ? statement.substring(0, 300) + "..." : statement;

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

  inputSection.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  errorSection.classList.add("hidden");

  statementInput.value = "";
  statementInput.focus();
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
  statementInput.focus();
});
