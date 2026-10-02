const API_BASE_URL = document.body.dataset.apiBase || "";
const REQUEST_TIMEOUT_MS = 15000;

const form = document.querySelector("#prediction-form");
const submitButton = document.querySelector("#submit-button");
const errorSummary = document.querySelector("#error-summary");
const errorSummaryMessage = document.querySelector("#error-summary-message");
const errorList = document.querySelector("#error-list");
const requestStatus = document.querySelector("#request-status");
const resultPanel = document.querySelector("#result-panel");
const probabilityElement = document.querySelector("#probability");
const predictionLabel = document.querySelector("#prediction-label");
const resultExplanation = document.querySelector("#result-explanation");
const scaleMarker = document.querySelector("#scale-marker");

const integerFields = new Set(["SeniorCitizen", "tenure"]);
const numberFields = new Set(["MonthlyCharges", "TotalCharges"]);
let activeController = null;
let requestSequence = 0;

function clearErrors() {
  errorSummary.hidden = true;
  errorList.replaceChildren();
  for (const field of form.elements) {
    if (!(field instanceof HTMLInputElement || field instanceof HTMLSelectElement)) continue;
    field.removeAttribute("aria-invalid");
    const error = document.querySelector(`#${CSS.escape(field.name)}-error`);
    if (error) error.textContent = "";
  }
}

function setFieldError(name, message) {
  const field = form.elements.namedItem(name);
  const error = document.querySelector(`#${CSS.escape(name)}-error`);
  if (!(field instanceof HTMLElement) || !error) return false;

  field.setAttribute("aria-invalid", "true");
  const describedBy = new Set((field.getAttribute("aria-describedby") || "").split(" ").filter(Boolean));
  describedBy.add(error.id);
  field.setAttribute("aria-describedby", [...describedBy].join(" "));
  error.textContent = message;

  if (errorList.childElementCount < 6) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = `#${name}`;
    link.textContent = `${field.labels?.[0]?.textContent || name}: ${message}`;
    item.append(link);
    errorList.append(item);
  }
  return true;
}

function showErrorSummary(message, focus = true) {
  errorSummaryMessage.textContent = message;
  errorSummary.hidden = false;
  if (focus) errorSummary.focus({ preventScroll: false });
}

function clientErrors() {
  const errors = [];
  for (const field of form.elements) {
    if (!(field instanceof HTMLInputElement || field instanceof HTMLSelectElement) || !field.name) continue;
    if (field.validity.valueMissing) {
      errors.push([field.name, "Choose or enter a value."]);
    } else if (field.validity.rangeUnderflow) {
      errors.push([field.name, "Enter zero or a positive value."]);
    } else if (field.validity.stepMismatch && field.name === "tenure") {
      errors.push([field.name, "Enter a whole number of months."]);
    } else if (field.validity.badInput || field.validity.stepMismatch) {
      errors.push([field.name, "Enter a valid number."]);
    }
  }
  return errors;
}

function payloadFromForm() {
  const data = Object.fromEntries(new FormData(form));
  for (const name of integerFields) data[name] = Number.parseInt(data[name], 10);
  for (const name of numberFields) data[name] = Number.parseFloat(data[name]);
  return data;
}

function readableApiMessage(error) {
  const raw = typeof error.msg === "string" ? error.msg : "Value is not accepted.";
  return raw.replace(/^Value error,\s*/i, "").replace(/^Input should be\s*/i, "Choose ");
}

function applyApiValidationErrors(detail) {
  let mapped = 0;
  for (const issue of detail) {
    const location = Array.isArray(issue.loc) ? issue.loc : [];
    const name = location.at(-1);
    if (typeof name === "string" && setFieldError(name, readableApiMessage(issue))) mapped += 1;
  }
  showErrorSummary(
    mapped ? "The API could not accept some details. Review the fields below and try again." : "The API could not validate this request. Review the details and try again."
  );
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Assessing customer…" : "Assess churn risk";
  form.setAttribute("aria-busy", String(isLoading));
}

function showResult(result) {
  const probability = result.churn_probability;
  const prediction = result.churn_prediction;
  if (!Number.isFinite(probability) || probability < 0 || probability > 1 || ![0, 1].includes(prediction)) {
    throw new Error("MALFORMED_RESPONSE");
  }

  const percentage = new Intl.NumberFormat(undefined, {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(probability);
  const churnPredicted = prediction === 1;

  probabilityElement.textContent = percentage;
  predictionLabel.textContent = churnPredicted ? "Churn predicted" : "No churn predicted";
  resultExplanation.textContent = churnPredicted
    ? "The model classifies this customer as likely to churn."
    : "The model classifies this customer as unlikely to churn.";
  scaleMarker.style.left = `${probability * 100}%`;
  resultPanel.classList.toggle("is-risk", churnPredicted);
  resultPanel.hidden = false;
  requestStatus.textContent = `Assessment complete. ${percentage} churn probability. ${predictionLabel.textContent}.`;
  resultPanel.focus({ preventScroll: false });
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) throw new Error("MALFORMED_RESPONSE");
  return response.json();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  activeController?.abort();
  const sequence = ++requestSequence;
  clearErrors();
  resultPanel.hidden = true;
  requestStatus.textContent = "";

  const errors = clientErrors();
  if (errors.length) {
    for (const [name, message] of errors) setFieldError(name, message);
    const remainder = errors.length - errorList.childElementCount;
    const remainderText = remainder > 0 ? ` The first ${errorList.childElementCount} are listed here; ${remainder} more are marked in the form.` : "";
    showErrorSummary(`${errors.length} fields need attention.${remainderText}`);
    return;
  }

  activeController = new AbortController();
  let didTimeout = false;
  const timeout = window.setTimeout(() => {
    didTimeout = true;
    activeController.abort();
  }, REQUEST_TIMEOUT_MS);
  setLoading(true);
  requestStatus.textContent = "Assessing customer churn risk.";

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(payloadFromForm()),
      signal: activeController.signal,
    });
    const body = await parseResponse(response);
    if (sequence !== requestSequence) return;

    if (!response.ok) {
      if (response.status === 422 && Array.isArray(body.detail)) {
        applyApiValidationErrors(body.detail);
      } else {
        showErrorSummary("The service could not complete the assessment. Your entries are still here—please try again.");
      }
      requestStatus.textContent = "Assessment failed.";
      return;
    }
    showResult(body);
  } catch (error) {
    if (sequence !== requestSequence) return;
    if (error.name === "AbortError" && !didTimeout) return;

    const message = error.message === "MALFORMED_RESPONSE"
      ? "The service returned an unexpected response. Your entries are still here—please try again."
      : didTimeout
        ? "The request took too long. Check the service and try again."
        : "The prediction service could not be reached. Check your connection and try again.";
    showErrorSummary(message);
    requestStatus.textContent = "Assessment failed.";
  } finally {
    window.clearTimeout(timeout);
    if (sequence === requestSequence) setLoading(false);
  }
});

form.addEventListener("input", (event) => {
  const field = event.target;
  if (!(field instanceof HTMLInputElement || field instanceof HTMLSelectElement)) return;
  field.removeAttribute("aria-invalid");
  const error = document.querySelector(`#${CSS.escape(field.name)}-error`);
  if (error) error.textContent = "";
});
