const API_BASE = "/api";
const PROJECT_INFO = {
  model: "QuickDraw CNN (20 classes)",
  stack: "FastAPI + Vanilla JavaScript",
};

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("statusText");
const robotBubble = document.getElementById("robotBubble");
const robotFace = document.getElementById("robotFace");
const bestGuess = document.getElementById("bestGuess");
const bestCard = document.getElementById("bestCard");
const latencyText = document.getElementById("latencyText");
const rankList = document.getElementById("rankList");
const classChips = document.getElementById("classChips");
const historyList = document.getElementById("historyList");
const latencyAvg = document.getElementById("latencyAvg");
const latencyMax = document.getElementById("latencyMax");
const modelStatusDot = document.getElementById("modelStatusDot");
const modelStatusText = document.getElementById("modelStatusText");

const toolBrush = document.getElementById("toolBrush");
const toolEraser = document.getElementById("toolEraser");
const lineWidthInput = document.getElementById("lineWidth");
const lineColorInput = document.getElementById("lineColor");
const clearBtn = document.getElementById("clearBtn");
const exportBtn = document.getElementById("exportBtn");

const watermarkId = document.getElementById("wmId");
const watermarkName = document.getElementById("wmName");

const state = {
  isDrawing: false,
  hasDrawing: false,
  dirtySinceLastPredict: false,
  isPredicting: false,
  tool: "brush",
  lineWidth: Number(lineWidthInput.value),
  lineColor: lineColorInput.value,
  lastTopLabel: null,
  stableCount: 0,
  lastCelebratedLabel: null,
  lastDisplaySize: null,
  lastDevicePixelRatio: null,
  predictionHistory: [],
  latencySamples: [],
  scrollLockedForTouch: false,
};

const LOGICAL_CANVAS_SIZE = 280;
let resizeTimer = null;

function setupWatermark() {
  watermarkId.textContent = `Model: ${PROJECT_INFO.model}`;
  watermarkName.textContent = `Stack: ${PROJECT_INFO.stack}`;
}

function snapshotCanvas() {
  const snap = document.createElement("canvas");
  snap.width = LOGICAL_CANVAS_SIZE;
  snap.height = LOGICAL_CANVAS_SIZE;
  const snapCtx = snap.getContext("2d");
  snapCtx.drawImage(canvas, 0, 0, LOGICAL_CANVAS_SIZE, LOGICAL_CANVAS_SIZE);
  return snap;
}

function setupCanvas({ preserveDrawing = false } = {}) {
  const displaySize = Math.min(canvas.parentElement.clientWidth - 8, 520);
  const ratio = window.devicePixelRatio || 1;

  const unchanged =
    state.lastDisplaySize === displaySize &&
    state.lastDevicePixelRatio === ratio;
  if (unchanged) return;

  const snap = preserveDrawing && state.hasDrawing ? snapshotCanvas() : null;

  canvas.style.width = `${displaySize}px`;
  canvas.style.height = `${displaySize}px`;

  canvas.width = Math.floor(LOGICAL_CANVAS_SIZE * ratio);
  canvas.height = Math.floor(LOGICAL_CANVAS_SIZE * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  resetCanvas();
  if (snap) {
    ctx.drawImage(snap, 0, 0, LOGICAL_CANVAS_SIZE, LOGICAL_CANVAS_SIZE);
    applyPenStyle();
  }

  state.lastDisplaySize = displaySize;
  state.lastDevicePixelRatio = ratio;
}

function resetCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, LOGICAL_CANVAS_SIZE, LOGICAL_CANVAS_SIZE);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  applyPenStyle();
}

function applyPenStyle() {
  ctx.lineWidth = state.lineWidth;
  ctx.strokeStyle = state.tool === "eraser" ? "white" : state.lineColor;
}

function eventToPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = LOGICAL_CANVAS_SIZE / rect.width;
  const scaleY = LOGICAL_CANVAS_SIZE / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function startDraw(event) {
  event.preventDefault();
  state.isDrawing = true;
  state.hasDrawing = true;
  state.dirtySinceLastPredict = true;
  if (event.pointerType === "touch" || event.pointerType === "pen") {
    document.documentElement.classList.add("drawing-lock");
    state.scrollLockedForTouch = true;
  }

  canvas.setPointerCapture(event.pointerId);
  applyPenStyle();

  const p = eventToPoint(event);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);

  statusText.textContent = "AI is observing your strokes...";
}

function draw(event) {
  if (!state.isDrawing) return;
  event.preventDefault();

  const p = eventToPoint(event);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  state.dirtySinceLastPredict = true;
}

function endDraw(event) {
  if (!state.isDrawing) return;
  event.preventDefault();
  state.isDrawing = false;
  if (state.scrollLockedForTouch) {
    document.documentElement.classList.remove("drawing-lock");
    state.scrollLockedForTouch = false;
  }
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

function setTool(tool) {
  state.tool = tool;
  toolBrush.classList.toggle("btn-solid", tool === "brush");
  toolEraser.classList.toggle("btn-solid", tool === "eraser");
  applyPenStyle();
}

function clearBoard() {
  resetCanvas();
  state.isDrawing = false;
  if (state.scrollLockedForTouch) {
    document.documentElement.classList.remove("drawing-lock");
    state.scrollLockedForTouch = false;
  }
  state.hasDrawing = false;
  state.dirtySinceLastPredict = false;
  state.lastTopLabel = null;
  state.stableCount = 0;
  state.lastCelebratedLabel = null;
  state.predictionHistory = [];
  state.latencySamples = [];

  bestGuess.textContent = "-";
  rankList.innerHTML = "";
  historyList.innerHTML = "";
  latencyText.textContent = "Latency: - ms";
  latencyAvg.textContent = "- ms";
  latencyMax.textContent = "- ms";
  statusText.textContent = "Waiting for your first stroke...";
  robotBubble.textContent = "I am ready to guess.";
}

function renderClassChips(classes) {
  classChips.innerHTML = "";
  classes.forEach((name) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = name;
    classChips.appendChild(chip);
  });
}

function renderRanking(predictions) {
  rankList.innerHTML = "";
  predictions.forEach((item, idx) => {
    const row = document.createElement("div");
    row.className = "rank-item";

    const name = document.createElement("div");
    name.className = "rank-name";
    name.textContent = `${idx + 1}. ${item.label}`;

    const bar = document.createElement("div");
    bar.className = "rank-bar";
    const fill = document.createElement("div");
    fill.className = "rank-fill";
    fill.style.width = `${Math.max(3, item.prob * 100)}%`;
    bar.appendChild(fill);

    const val = document.createElement("div");
    val.className = "rank-val";
    val.textContent = `${(item.prob * 100).toFixed(1)}%`;

    row.appendChild(name);
    row.appendChild(bar);
    row.appendChild(val);
    rankList.appendChild(row);
  });
}

function playCelebrate() {
  bestCard.classList.remove("guess-celebrate");
  robotFace.classList.remove("robot-cheer");

  void bestCard.offsetWidth;
  bestCard.classList.add("guess-celebrate");
  robotFace.classList.add("robot-cheer");

  setTimeout(() => {
    bestCard.classList.remove("guess-celebrate");
    robotFace.classList.remove("robot-cheer");
  }, 1000);
}

function addArticle(word) {
  if (!word) return word;
  const first = word[0].toLowerCase();
  const vowels = ["a", "e", "i", "o", "u"];
  return `${vowels.includes(first) ? "an" : "a"} ${word}`;
}

function bubbleText(topLabel, topProb, guessedNow, secondLabel) {
  const labelWithArticle = addArticle(topLabel);
  if (guessedNow || topProb >= 0.9) {
    return `This looks very much like ${labelWithArticle}.`;
  }
  if (topProb >= 0.75) {
    return `I think this is ${labelWithArticle}!`;
  }
  if (topProb >= 0.65) {
    return `I am guessing ${topLabel} now.`;
  }
  if (topProb >= 0.4) {
    return `Not sure yet... but ${topLabel} is most likely.`;
  }
  if (secondLabel) {
    return `Still uncertain. ${topLabel} leads, ${secondLabel} is close behind.`;
  }
  return `Still uncertain, but ${topLabel} is leading for now.`;
}

function updateLatencyStats(currentLatencyMs) {
  state.latencySamples.push(currentLatencyMs);
  if (state.latencySamples.length > 200) {
    state.latencySamples.shift();
  }
  const sum = state.latencySamples.reduce((acc, item) => acc + item, 0);
  const avg = sum / state.latencySamples.length;
  const max = Math.max(...state.latencySamples);
  latencyAvg.textContent = `${avg.toFixed(1)} ms`;
  latencyMax.textContent = `${max.toFixed(1)} ms`;
}

function pushPredictionHistory(topLabel, topProb) {
  const now = new Date();
  const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
  state.predictionHistory.unshift({
    time,
    label: topLabel,
    prob: topProb,
  });
  if (state.predictionHistory.length > 10) {
    state.predictionHistory.pop();
  }

  historyList.innerHTML = "";
  state.predictionHistory.forEach((item) => {
    const row = document.createElement("div");
    row.className = "history-item";
    row.innerHTML = `
      <div class="history-time">${item.time}</div>
      <div class="history-label">${item.label}</div>
      <div class="history-prob">${(item.prob * 100).toFixed(1)}%</div>
    `;
    historyList.appendChild(row);
  });
}

function updateGuessState(predictions) {
  if (!predictions.length) {
    state.lastTopLabel = null;
    state.stableCount = 0;
    return false;
  }

  const top = predictions[0];
  const aboveThreshold = top.prob >= 0.65;

  if (!aboveThreshold) {
    state.lastTopLabel = null;
    state.stableCount = 0;
    return false;
  }

  if (state.lastTopLabel === top.label) {
    state.stableCount += 1;
  } else {
    state.lastTopLabel = top.label;
    state.stableCount = 1;
  }

  return state.stableCount >= 2;
}

async function sendPrediction() {
  if (!state.hasDrawing || !state.dirtySinceLastPredict || state.isPredicting) return;

  state.isPredicting = true;
  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_b64: canvas.toDataURL("image/png"),
        top_k: 5,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    state.dirtySinceLastPredict = false;
    latencyText.textContent = `Latency: ${data.latency_ms.toFixed(1)} ms`;
    updateLatencyStats(data.latency_ms);

    if (data.is_blank) {
      bestGuess.textContent = "-";
      rankList.innerHTML = "";
      statusText.textContent = "Please start drawing.";
      robotBubble.textContent = "I cannot see enough ink yet.";
      state.lastTopLabel = null;
      state.stableCount = 0;
      return;
    }

    renderRanking(data.predictions);
    const top = data.predictions[0];
    const second = data.predictions[1];
    bestGuess.textContent = top ? top.label : "-";
    pushPredictionHistory(top.label, top.prob);

    const guessedNow = updateGuessState(data.predictions);
    if (guessedNow) {
      statusText.textContent = `Guessed: ${top.label}`;
      robotBubble.textContent = bubbleText(top.label, top.prob, true, second?.label);

      if (state.lastCelebratedLabel !== top.label) {
        playCelebrate();
        state.lastCelebratedLabel = top.label;
      }
    } else {
      statusText.textContent = `Thinking... top guess is ${top.label}`;
      robotBubble.textContent = bubbleText(top.label, top.prob, false, second?.label);
    }
  } catch (err) {
    statusText.textContent = `Error: ${err.message}`;
    robotBubble.textContent = "I hit an API error. Check backend logs.";
  } finally {
    state.isPredicting = false;
  }
}

async function loadClassesAndHealth() {
  try {
    const [healthRes, classesRes] = await Promise.all([
      fetch(`${API_BASE}/health`),
      fetch(`${API_BASE}/classes`),
    ]);

    const healthData = await healthRes.json();
    const classesData = await classesRes.json();

    if (!healthData.model_loaded) {
      modelStatusDot.classList.remove("status-on");
      modelStatusDot.classList.add("status-off");
      modelStatusText.textContent = "Model not loaded";
      statusText.textContent = "Model is not loaded. Run python ml/train.py first.";
      robotBubble.textContent = "I need training before I can guess.";
    } else {
      modelStatusDot.classList.remove("status-off");
      modelStatusDot.classList.add("status-on");
      modelStatusText.textContent = "Model loaded";
    }

    renderClassChips(classesData.classes || []);
  } catch (err) {
    modelStatusDot.classList.remove("status-on");
    modelStatusDot.classList.add("status-off");
    modelStatusText.textContent = "Backend unreachable";
    statusText.textContent = `Startup check failed: ${err.message}`;
    robotBubble.textContent = "Backend not reachable.";
  }
}

function exportFailureSample() {
  const now = new Date();
  const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}_${String(now.getHours()).padStart(2, "0")}${String(now.getMinutes()).padStart(2, "0")}${String(now.getSeconds()).padStart(2, "0")}`;
  const topLabel = bestGuess.textContent && bestGuess.textContent !== "-" ? bestGuess.textContent : "unknown";
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = `failure_sample_${topLabel}_${stamp}.png`;
  link.click();
}

function bindEvents() {
  canvas.addEventListener("pointerdown", startDraw);
  canvas.addEventListener("pointermove", draw);
  canvas.addEventListener("pointerup", endDraw);
  canvas.addEventListener("pointercancel", endDraw);
  canvas.addEventListener("pointerleave", endDraw);

  toolBrush.addEventListener("click", () => setTool("brush"));
  toolEraser.addEventListener("click", () => setTool("eraser"));
  clearBtn.addEventListener("click", clearBoard);
  exportBtn.addEventListener("click", exportFailureSample);

  lineWidthInput.addEventListener("input", () => {
    state.lineWidth = Number(lineWidthInput.value);
    applyPenStyle();
  });

  lineColorInput.addEventListener("input", () => {
    state.lineColor = lineColorInput.value;
    if (state.tool === "brush") applyPenStyle();
  });

  const handleResize = () => {
    if (state.isDrawing) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      setupCanvas({ preserveDrawing: true });
    }, 120);
  };

  window.addEventListener("resize", handleResize);
  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", handleResize);
  }
}

function main() {
  setupWatermark();
  setupCanvas();
  bindEvents();
  loadClassesAndHealth();
  setInterval(sendPrediction, 500);
}

main();
