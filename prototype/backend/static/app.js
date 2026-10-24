/**
 * MessMate — Open-World Dietary Intelligence
 * Client Controller (Pure Vanilla JS — Zero Dependencies)
 * Adheres strictly to Invariants I1 through I10.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    token: localStorage.getItem("messmate_token"),
    user: null,
    diningContext: "personal_home",
    selectedFile: null,
    activeMeal: null,
    pollTimer: null,
  };

  // DOM Elements
  const el = {
    userDisplay: document.getElementById("user-display"),
    authStatus: document.getElementById("auth-status"),
    btnQuickAuth: document.getElementById("btn-quick-auth"),
    navTabs: document.querySelectorAll(".nav-tab"),
    tabPanes: document.querySelectorAll(".tab-pane"),
    budgetCount: document.getElementById("budget-count"),
    ctxHome: document.getElementById("ctx-home"),
    ctxAway: document.getElementById("ctx-away"),
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    viewfinderPrompt: document.getElementById("viewfinder-prompt"),
    imagePreview: document.getElementById("image-preview"),
    btnSelectFile: document.getElementById("btn-select-file"),
    btnSampleMeal: document.getElementById("btn-sample-meal"),
    btnSubmitMeal: document.getElementById("btn-submit-meal"),
    analysisLoader: document.getElementById("analysis-loader"),
    mealResultCard: document.getElementById("meal-result-card"),
    resultStatus: document.getElementById("result-status"),
    resultContext: document.getElementById("result-context"),
    resultTime: document.getElementById("result-time"),
    intervalNumbers: document.getElementById("interval-numbers"),
    uncertaintyRange: document.getElementById("uncertainty-range"),
    uncertaintyMarker: document.getElementById("uncertainty-marker"),
    meanCalories: document.getElementById("mean-calories"),
    sdCalories: document.getElementById("sd-calories"),
    macroProtein: document.getElementById("macro-protein"),
    macroFat: document.getElementById("macro-fat"),
    macroCarb: document.getElementById("macro-carb"),
    macroFibre: document.getElementById("macro-fibre"),
    questionBox: document.getElementById("question-box"),
    qBudgetRem: document.getElementById("q-budget-rem"),
    questionPrompt: document.getElementById("question-prompt"),
    questionOptions: document.getElementById("question-options"),
    itemsList: document.getElementById("items-list"),
    refillChips: document.getElementById("refill-chips"),
    // Onboarding
    btnSaveOnboarding: document.getElementById("btn-save-onboarding"),
    onboardingFeedback: document.getElementById("onboarding-feedback"),
    // Today
    todayDateBadge: document.getElementById("today-date-badge"),
    dailyTotalKcal: document.getElementById("daily-total-kcal"),
    dailyMealsCount: document.getElementById("daily-meals-count"),
    dailyQuestionsUsed: document.getElementById("daily-questions-used"),
    todayMealsList: document.getElementById("today-meals-list"),
    // Digest
    digestDaysLogged: document.getElementById("digest-days-logged"),
    digestMeanKcal: document.getElementById("digest-mean-kcal"),
    gapProteinVal: document.getElementById("gap-protein-val"),
    gapFibreVal: document.getElementById("gap-fibre-val"),
    progressProtein: document.getElementById("progress-protein"),
    progressFibre: document.getElementById("progress-fibre"),
    swapsContainer: document.getElementById("swaps-container"),
  };

  function getLocalDateString() {
    const now = new Date();
    const y = now.getFullYear();
    const m = String(now.getMonth() + 1).padStart(2, "0");
    const d = String(now.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
  }

  // API Client Helper with auto-retry on 401
  async function api(endpoint, options = {}, allowRetry = true) {
    const headers = { ...options.headers };
    if (state.token) {
      headers["Authorization"] = `Bearer ${state.token}`;
    }
    const config = {
      ...options,
      headers,
    };
    const res = await fetch(endpoint, config);
    if (res.status === 401 && allowRetry) {
      state.token = null;
      localStorage.removeItem("messmate_token");
      const reauthed = await quickDemoAuth();
      if (reauthed) {
        return api(endpoint, options, false);
      }
    }
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      const err = new Error(errBody.detail || `Request failed with status ${res.status}`);
      err.status = res.status;
      err.body = errBody;
      throw err;
    }
    return res.json();
  }

  // --- 1. Authentication ---
  let authInFlight = false;
  let authRetryCount = 0;

  async function checkAuth() {
    if (!state.token) {
      updateAuthUI(false, "Connecting...");
      return quickDemoAuth();
    }
    try {
      const profile = await api("/v1/me/profile", {}, false);
      state.user = profile;
      authRetryCount = 0;
      updateAuthUI(true, `Active (BMR: ${Math.round(profile.bmr)} kcal)`);
      refreshBudget();
      return true;
    } catch (e) {
      console.warn("Profile check failed:", e);
      if (e.status === 401 && authRetryCount < 2) {
        authRetryCount++;
        state.token = null;
        localStorage.removeItem("messmate_token");
        updateAuthUI(false, "Session refreshing...");
        return quickDemoAuth();
      } else {
        updateAuthUI(false, "Offline / Demo");
      }
      return false;
    }
  }

  function updateAuthUI(isAuthed, label) {
    if (isAuthed) {
      el.authStatus.classList.add("authenticated");
      el.userDisplay.textContent = label;
      el.btnQuickAuth.textContent = "Switch User";
    } else {
      el.authStatus.classList.remove("authenticated");
      el.userDisplay.textContent = label;
      el.btnQuickAuth.textContent = "Quick Demo Sign-In";
    }
  }

  async function quickDemoAuth() {
    if (authInFlight) return false;
    authInFlight = true;
    const demoEmail = "demo@messmate.app";
    const demoPass = "MessMate123!";

    try {
      // 1. Try login
      const res = await fetch("/v1/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: demoEmail, password: demoPass }),
      });
      if (res.ok) {
        const data = await res.json();
        state.token = data.access_token;
        localStorage.setItem("messmate_token", data.access_token);
        updateAuthUI(true, "Active");
        refreshBudget();
        showToast("Connected to MessMate server!");
        return true;
      }

      // 2. If login fails, register
      const regRes = await fetch("/v1/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: demoEmail,
          password: demoPass,
          birth_year: 1995,
          sex: "other",
          study_arm: "B_adaptive",
        }),
      });
      if (regRes.ok) {
        const regData = await regRes.json();
        state.token = regData.access_token;
        localStorage.setItem("messmate_token", regData.access_token);
        await api("/v1/me/consents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ purpose: "core_logging", granted: true }),
        }, false).catch(() => {});
        updateAuthUI(true, "Active");
        refreshBudget();
        showToast("Demo profile created & connected!");
        return true;
      }
    } catch (err) {
      console.warn("Auth attempt error:", err);
      showToast("Backend connection waiting...");
    } finally {
      authInFlight = false;
    }
    return false;
  }

  // --- 2. Navigation Tabs ---
  function initTabs() {
    el.navTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const targetId = tab.dataset.tab;
        el.navTabs.forEach((t) => t.classList.remove("active"));
        el.tabPanes.forEach((p) => p.classList.remove("active"));

        tab.classList.add("active");
        const pane = document.getElementById(targetId);
        if (pane) pane.classList.add("active");

        if (targetId === "tab-today") loadTodayRollup();
        if (targetId === "tab-digest") loadSundayDigest();
      });
    });
  }

  // --- 3. Context Gate (§6.1.2) ---
  function initContextGate() {
    el.ctxHome.addEventListener("click", () => {
      state.diningContext = "personal_home";
      el.ctxHome.classList.add("active");
      el.ctxAway.classList.remove("active");
    });
    el.ctxAway.addEventListener("click", () => {
      state.diningContext = "commercial";
      el.ctxAway.classList.add("active");
      el.ctxHome.classList.remove("active");
    });
  }

  // --- 4. Camera & Image Capture ---
  function initCapture() {
    el.btnSelectFile.addEventListener("click", () => el.fileInput.click());
    if (el.btnSampleMeal) {
      el.btnSampleMeal.addEventListener("click", async () => {
        try {
          const res = await fetch("/static/sample_thali.jpg?v=" + Date.now());
          const blob = await res.blob();
          const file = new File([blob], "sample_thali.jpg", { type: "image/jpeg" });
          handleFileSelection(file);
          showToast("Sample North Indian Thali plate loaded!");
        } catch (err) {
          showToast("Could not load sample plate: " + err.message);
        }
      });
    }
    el.imagePreview.addEventListener("click", () => el.fileInput.click());
    el.dropZone.addEventListener("click", (e) => {
      if (
        e.target !== el.btnSubmitMeal &&
        e.target !== el.btnSelectFile &&
        e.target !== el.btnSampleMeal
      ) {
        el.fileInput.click();
      }
    });

    el.fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) handleFileSelection(file);
    });

    // Drag and Drop
    el.dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      el.dropZone.style.borderColor = "var(--accent-primary)";
    });
    el.dropZone.addEventListener("dragleave", () => {
      el.dropZone.style.borderColor = "var(--border-subtle)";
    });
    el.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      el.dropZone.style.borderColor = "var(--border-subtle)";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    el.btnSubmitMeal.addEventListener("click", submitMeal);

    // Expose helpers for automated testing and deep-link resumption
    window.renderMealResult = renderMealResult;
    window.pollMealStatus = pollMealStatus;
    window.loadMealById = loadMealById;
  }

  // Client-side hardware canvas downsampling (< 1400px, WebP/JPEG < 200KB)
  async function compressImageForUpload(file, maxDimension = 1400, quality = 0.82) {
    if (!file || !file.type.startsWith("image/") || file.type === "image/svg+xml" || file.size <= 400 * 1024) {
      return file;
    }
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          let { width, height } = img;
          if (width > maxDimension || height > maxDimension) {
            if (width > height) {
              height = Math.round((height * maxDimension) / width);
              width = maxDimension;
            } else {
              width = Math.round((width * maxDimension) / height);
              height = maxDimension;
            }
          }
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0, width, height);
          canvas.toBlob(
            (blob) => {
              if (blob && blob.size < file.size) {
                const compressed = new File([blob], file.name.replace(/\.[^/.]+$/, "") + ".jpg", {
                  type: "image/jpeg",
                  lastModified: Date.now(),
                });
                resolve(compressed);
              } else {
                resolve(file);
              }
            },
            "image/jpeg",
            quality
          );
        };
        img.onerror = () => resolve(file);
        img.src = e.target.result;
      };
      reader.onerror = () => resolve(file);
      reader.readAsDataURL(file);
    });
  }

  function handleFileSelection(file) {
    state.selectedFile = file;
    const url = URL.createObjectURL(file);
    el.imagePreview.src = url;
    el.imagePreview.classList.remove("hidden");
    el.viewfinderPrompt.classList.add("hidden");
    el.btnSubmitMeal.disabled = false;
  }

  function getMealSlot() {
    const hour = new Date().getHours();
    if (hour >= 6 && hour < 11) return "breakfast";
    if (hour >= 11 && hour < 16) return "lunch";
    if (hour >= 16 && hour < 19) return "snack";
    return "dinner";
  }

  async function submitMeal() {
    if (!state.token) {
      await quickDemoAuth();
    }

    el.btnSubmitMeal.disabled = true;
    el.analysisLoader.classList.remove("hidden");
    el.mealResultCard.classList.add("hidden");

    let uploadFile = state.selectedFile;
    if (uploadFile && uploadFile.size > 400 * 1024) {
      showToast("Optimizing plate image for instant analysis...");
      uploadFile = await compressImageForUpload(uploadFile);
    }

    const formData = new FormData();
    if (uploadFile) {
      formData.append("file", uploadFile);
    }
    formData.append("context", state.diningContext);
    formData.append("meal_slot", getMealSlot());
    formData.append("captured_at", new Date().toISOString());

    try {
      // Initiate upload
      const uploadRes = await api("/v1/meals", {
        method: "POST",
        body: formData,
      });

      const mealId = uploadRes.meal_id;
      localStorage.setItem("messmate_active_meal_id", mealId);
      // Start polling for results
      pollMealStatus(mealId);
    } catch (err) {
      el.analysisLoader.classList.add("hidden");
      el.btnSubmitMeal.disabled = false;
      showToast("Failed to upload meal: " + err.message);
    }
  }

  function pollMealStatus(mealId) {
    if (state.pollTimer) clearInterval(state.pollTimer);
    let attempts = 0;
    const maxAttempts = 120; // 60s timeout for cloud multimodal perception

    state.pollTimer = setInterval(async () => {
      attempts++;
      const elapsedSec = Math.round(attempts * 0.5);
      const loaderTitle = el.analysisLoader.querySelector(".loader-title");
      if (loaderTitle) {
        loaderTitle.textContent = `Analyzing Plate & Tableware... (${elapsedSec}s)`;
      }

      if (attempts >= maxAttempts) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        el.analysisLoader.classList.add("hidden");
        el.btnSubmitMeal.disabled = false;
        showToast("Meal analysis timed out. Please retry with clear lighting.");
        return;
      }

      try {
        const meal = await api(`/v1/meals/${mealId}`);
        const isTerminalSuccess =
          meal.status === "analyzed" ||
          meal.status === "ready_silent" ||
          meal.status === "ready_with_question";

        if (isTerminalSuccess || meal.status === "failed") {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          el.analysisLoader.classList.add("hidden");
          el.btnSubmitMeal.disabled = false;

          if (isTerminalSuccess) {
            localStorage.setItem("messmate_active_meal_id", mealId);
            renderMealResult(meal);
          } else {
            showToast("Meal analysis failed. Please retry with clear lighting.");
          }
        }
      } catch (e) {
        console.error("Poll error:", e);
      }
    }, 500);
  }

  async function loadMealById(mealId) {
    if (!mealId) return;
    try {
      const meal = await api(`/v1/meals/${mealId}`);
      if (meal && (meal.status === "analyzed" || meal.status === "ready_silent" || meal.status === "ready_with_question")) {
        localStorage.setItem("messmate_active_meal_id", mealId);
        renderMealResult(meal);
        if (el.navTabs && el.navTabs[0]) {
          el.navTabs[0].click();
        }
        showToast("Active meal restored.");
      }
    } catch (e) {
      console.warn("Could not restore meal by id:", mealId, e);
    }
  }

  // --- 5. Render Meal Result ---
  function renderMealResult(meal) {
    state.activeMeal = meal;
    el.mealResultCard.classList.remove("hidden");

    // Badges & timestamp
    el.resultStatus.textContent = meal.status.toUpperCase();
    el.resultContext.textContent = meal.context === "personal_home" ? "Home (Calibrated)" : "Away (Prior Frozen)";
    el.resultTime.textContent = new Date(meal.captured_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    // Conformal Interval [p10, p90]
    if (meal.total) {
      el.intervalNumbers.textContent = `${Math.round(meal.total.kcal_p10)} – ${Math.round(meal.total.kcal_p90)}`;
      el.meanCalories.textContent = Math.round(meal.total.kcal_mean);
      el.sdCalories.textContent = Math.round(meal.total.kcal_sd);

      el.macroProtein.textContent = `${Math.round(meal.total.protein_g)}g`;
      el.macroFat.textContent = `${Math.round(meal.total.fat_g)}g`;
      el.macroCarb.textContent = `${Math.round(meal.total.carb_g)}g`;
      el.macroFibre.textContent = `${Math.round(meal.total.fibre_g)}g`;

      // Position tactile Uncertainty Range Gauge
      if (el.uncertaintyRange && el.uncertaintyMarker) {
        const p10 = meal.total.kcal_p10 || 0;
        const p90 = meal.total.kcal_p90 || 0;
        const mean = meal.total.kcal_mean || 0;
        const maxScale = Math.max(2200, p90 * 1.18);
        const leftPct = Math.max(2, Math.min(90, (p10 / maxScale) * 100));
        const rightPct = Math.max(leftPct + 4, Math.min(98, (p90 / maxScale) * 100));
        const widthPct = Math.max(4, rightPct - leftPct);
        const meanPct = Math.max(2, Math.min(98, (mean / maxScale) * 100));

        el.uncertaintyRange.style.left = `${leftPct}%`;
        el.uncertaintyRange.style.width = `${widthPct}%`;
        el.uncertaintyMarker.style.left = `${meanPct}%`;
      }
    }

    // Budget remaining & Reverse Friction Curve Stage
    if (meal.budget) {
      let stageName = "Week 1 Learning";
      if (meal.budget.total <= 1 && meal.budget.total > 0) stageName = "Week 2+ Refining";
      else if (meal.budget.total === 0) stageName = "Steady-State (Silent)";
      el.budgetCount.textContent = `${stageName} • ${meal.budget.remaining}/${meal.budget.total} left`;
      el.qBudgetRem.textContent = meal.budget.remaining;
    }

    // Render Question Shade if present and unanswered (Invariant I3 & I4)
    if (meal.question && !meal.question.skipped) {
      el.questionBox.classList.remove("hidden");
      el.questionPrompt.textContent = meal.question.prompt;
      el.questionOptions.innerHTML = "";

      // Invariant I4: Max 4 options, discrete buttons
      (meal.question.options || []).slice(0, 4).forEach((opt) => {
        const btn = document.createElement("button");
        btn.className = "opt-btn";
        btn.textContent = opt.label;
        btn.addEventListener("click", function() { answerQuestion(meal.meal_id, opt.value, this); });
        el.questionOptions.appendChild(btn);
      });

      // Invariant I4: Frictionless 1-tap Skip button
      const skipBtn = document.createElement("button");
      skipBtn.className = "opt-btn btn-skip";
      skipBtn.style.gridColumn = "1 / -1";
      skipBtn.style.background = "transparent";
      skipBtn.style.border = "1px dashed rgba(255, 255, 255, 0.25)";
      skipBtn.style.color = "var(--text-muted)";
      skipBtn.style.fontSize = "12px";
      skipBtn.textContent = "Skip clarification (keep current estimate)";
      skipBtn.addEventListener("click", function() { answerQuestion(meal.meal_id, null, this, true); });
      el.questionOptions.appendChild(skipBtn);

      // Invariant I4: Schedule native T+20 min delayed notification if running in Android app
      if (window.MessMateNative && window.MessMateNative.scheduleDelayedNotification) {
        try {
          window.MessMateNative.scheduleDelayedNotification(
            meal.meal_id,
            meal.question.prompt,
            JSON.stringify(meal.question.options || []),
            20 * 60 // 20 minutes
          );
        } catch (_) {}
      }
    } else {
      el.questionBox.classList.add("hidden");
    }

    // Identified items list
    el.itemsList.innerHTML = "";
    (meal.items || []).forEach((item) => {
      const row = document.createElement("div");
      row.className = "item-row";
      row.innerHTML = `
        <div class="item-name">${escapeHtml(item.dish_label)}</div>
        <div class="item-meta">
          <span class="item-grams">${Math.round(item.grams_mean || 0)}g (±${Math.round(item.grams_sd || 0)})</span>
          <span class="item-kcal">${Math.round(item.kcal_mean || 0)} kcal</span>
        </div>
      `;
      el.itemsList.appendChild(row);
    });

    // Dynamic Refill Chips (Invariant I5)
    setupRefillChips(meal.items);
  }

  async function answerQuestion(mealId, answerValue, sourceBtn, isSkip = false) {
    const allBtns = el.questionOptions.querySelectorAll("button");
    allBtns.forEach((b) => (b.disabled = true));
    if (sourceBtn) sourceBtn.textContent = isSkip ? "Skipping..." : "Recording...";

    try {
      await api(`/v1/meals/${mealId}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer: answerValue, skipped: isSkip }),
      });
      el.questionBox.classList.add("hidden");
      showToast(isSkip ? "Clarification skipped." : "Serving recorded. Calorie range narrowed.");
      // Refresh meal to show narrowed interval
      const updatedMeal = await api(`/v1/meals/${mealId}`);
      renderMealResult(updatedMeal);
      refreshBudget();
    } catch (err) {
      if (err.message && err.message.includes("already been answered")) {
        el.questionBox.classList.add("hidden");
      } else {
        showToast("Could not submit answer: " + (err.message || "Network error"));
        allBtns.forEach((b) => (b.disabled = false));
      }
    }
  }

  function setupRefillChips(items) {
    el.refillChips.innerHTML = "";
    if (!items || items.length === 0) {
      // Cold-start staple presets (Invariant I5: 1-tap discrete additions)
      const staples = [
        { label: "+1 Roti", text: "1 roti" },
        { label: "+1 Katori Dal", text: "1 katori dal tadka" },
        { label: "+½ Plate Rice", text: "half plate cooked rice" },
        { label: "+1 Cup Curd", text: "1 cup curd" },
      ];
      staples.forEach((s) => {
        const btn = document.createElement("button");
        btn.className = "chip-btn";
        btn.textContent = s.label;
        btn.title = `1-tap log ${s.label}`;
        btn.addEventListener("click", () => handleStapleRefill(s.label, s.text));
        el.refillChips.appendChild(btn);
      });
      return;
    }

    items.forEach((item) => {
      // 1-serving chip
      const btn1 = document.createElement("button");
      btn1.className = "chip-btn";
      btn1.textContent = `+1 ${item.dish_label}`;
      btn1.addEventListener("click", function() { handleRefill(item, 1.0, this); });
      el.refillChips.appendChild(btn1);

      // If item is a bowl/curry/rice, also provide +½ chip
      const u = (item.unit || "").toLowerCase();
      if (u.includes("bowl") || u.includes("cup") || (item.grams_mean && item.grams_mean > 120)) {
        const btnHalf = document.createElement("button");
        btnHalf.className = "chip-btn";
        btnHalf.textContent = `+½ ${item.dish_label}`;
        btnHalf.addEventListener("click", function() { handleRefill(item, 0.5, this); });
        el.refillChips.appendChild(btnHalf);
      }
    });
  }

  async function handleStapleRefill(label, text) {
    if (!state.token) await quickDemoAuth();
    try {
      showToast(`Logging ${label}...`);
      const meal = await api("/v1/meals/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          meal_slot: "snack",
          context: state.diningContext || "personal_home",
          source: "text",
        }),
      });
      showToast(`Added ${label}!`);
      renderMealResult(meal);
      refreshBudget();
    } catch (err) {
      showToast("Refill logging failed: " + err.message);
    }
  }

  async function handleRefill(item, delta, btnElement) {
    if (btnElement) {
      btnElement.disabled = true;
      btnElement.style.opacity = "0.5";
    }
    try {
      const currentCount = item.unit_count || 1.0;
      const nextCount = Math.round((currentCount + delta) * 10) / 10;
      item.unit_count = nextCount;

      await api(`/v1/meal-items/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ unit_count: nextCount, is_refill: true }),
      });
      const label = delta === 0.5 ? "½" : delta;
      showToast(`Added +${label} ${item.dish_label}!`);
      // Re-fetch meal
      if (state.activeMeal) {
        const updated = await api(`/v1/meals/${state.activeMeal.meal_id}`);
        renderMealResult(updated);
        refreshBudget();
      }
    } catch (e) {
      showToast("Refill update failed: " + e.message);
    } finally {
      if (btnElement) {
        btnElement.disabled = false;
        btnElement.style.opacity = "1";
      }
    }
  }

  // --- 6. Visual Kitchen Mirror Onboarding ---
  function initOnboarding() {
    // Archetype selection active styling
    document.querySelectorAll(".archetype-card").forEach((card) => {
      card.addEventListener("click", () => {
        const radio = card.querySelector('input[type="radio"]');
        if (radio) {
          const groupName = radio.name;
          document.querySelectorAll(`input[name="${groupName}"]`).forEach((r) => {
            r.closest(".archetype-card").classList.remove("active");
          });
          card.classList.add("active");
          radio.checked = true;
        }
      });
    });

    el.btnSaveOnboarding.addEventListener("click", async () => {
      if (!state.token) await quickDemoAuth();

      const bowlRadio = document.querySelector('input[name="bowl-size"]:checked');
      const plateRadio = document.querySelector('input[name="plate-size"]:checked');
      const mugRadio = document.querySelector('input[name="mug-size"]:checked');

      const archetypes = [
        {
          unit: bowlRadio.dataset.unit || "standard_bowl",
          volume_ml: parseFloat(bowlRadio.value),
          label: "Primary Bowl",
        },
        {
          unit: plateRadio.dataset.unit || "dinner_plate",
          volume_ml: parseFloat(plateRadio.value),
          label: "Dinner Plate",
        },
        {
          unit: mugRadio.dataset.unit || "cup_mug",
          volume_ml: parseFloat(mugRadio.value),
          label: "Primary Mug",
        },
      ];

      try {
        el.btnSaveOnboarding.disabled = true;
        const res = await api("/v1/me/onboarding/tableware", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ archetypes }),
        });

        el.onboardingFeedback.className = "feedback-msg success";
        el.onboardingFeedback.textContent = res.message || "Tableware successfully calibrated!";
        el.onboardingFeedback.classList.remove("hidden");
        showToast("Tableware saved to personal memory.");
      } catch (err) {
        el.onboardingFeedback.className = "feedback-msg error";
        el.onboardingFeedback.textContent = "Calibration failed: " + err.message;
        el.onboardingFeedback.classList.remove("hidden");
      } finally {
        el.btnSaveOnboarding.disabled = false;
      }
    });
  }

  // --- 7. Today's Rollup ---
  async function loadTodayRollup() {
    if (!state.token) return;
    const today = new Date().toISOString().split("T")[0];
    el.todayDateBadge.textContent = today;

    try {
      const data = await api(`/v1/days/${today}`);
      el.dailyTotalKcal.textContent = Math.round(data.total_kcal_mean);
      el.dailyMealsCount.textContent = data.meal_count;
      el.dailyQuestionsUsed.textContent = `${data.budget.used} / ${data.budget.total}`;

      if (!data.meals || data.meals.length === 0) {
        el.todayMealsList.innerHTML = `<p class="empty-state">No meals logged yet today. Snap a photo in the first tab to begin!</p>`;
        return;
      }

      el.todayMealsList.innerHTML = "";
      data.meals.forEach((m) => {
        const itemNames = (m.items || []).map((i) => i.dish_label).join(", ") || "Analyzed Plate";
        const div = document.createElement("div");
        div.className = "today-meal-card";
        div.innerHTML = `
          <div class="today-meal-info">
            <strong>${escapeHtml(itemNames)}</strong>
            <br>
            <span>${new Date(m.captured_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} &bull; ${m.context === "personal_home" ? "Home" : "Away"}</span>
          </div>
          <div class="item-kcal">${Math.round(m.total ? m.total.kcal_mean : 0)} kcal</div>
        `;
        el.todayMealsList.appendChild(div);
      });
    } catch (err) {
      console.warn("Could not load day rollup:", err);
    }
  }

  async function refreshBudget() {
    const today = new Date().toISOString().split("T")[0];
    try {
      const data = await api(`/v1/days/${today}`);
      if (data && data.budget) {
        let stageName = "Week 1 Learning";
        if (data.meal_count !== undefined && data.meal_count <= 3 && data.budget.total >= 3) {
          stageName = "Day 1 Mirror";
        } else if (data.budget.total <= 1 && data.budget.total > 0) {
          stageName = "Week 2+ Refining";
        } else if (data.budget.total === 0) {
          stageName = "Steady-State (Silent)";
        }
        el.budgetCount.textContent = `${stageName} • ${data.budget.remaining}/${data.budget.total} left`;
        if (el.qBudgetRem) el.qBudgetRem.textContent = data.budget.remaining;
      }
    } catch (e) {}
  }

  // --- 8. Sunday Weekly Digest ---
  function getISOWeek() {
    const d = new Date();
    d.setHours(0, 0, 0, 0);
    d.setDate(d.getDate() + 4 - (d.getDay() || 7));
    const yearStart = new Date(d.getFullYear(), 0, 1);
    const weekNo = Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
    return `${d.getFullYear()}-W${String(weekNo).padStart(2, "0")}`;
  }

  async function loadSundayDigest() {
    if (!state.token) return;
    const isoWeek = getISOWeek();

    try {
      const digest = await api(`/v1/weeks/${isoWeek}`);
      el.digestDaysLogged.textContent = digest.days_logged || 0;
      el.digestMeanKcal.textContent = Math.round(digest.mean_daily_kcal || 0);

      // Protein progress (Daily Goal: 54g/day)
      const proteinLogged = Math.round(digest.mean_daily_protein_g || 0);
      const proteinGoal = 54;
      const proteinPct = Math.min(100, Math.round((proteinLogged / proteinGoal) * 100));
      el.gapProteinVal.textContent = `${proteinLogged}g / ${proteinGoal}g avg (${proteinPct}%)`;
      el.progressProtein.style.width = `${proteinPct}%`;

      // Fibre progress (Daily Goal: 30g/day)
      const fibreLogged = Math.round(digest.mean_daily_fibre_g || 0);
      const fibreGoal = 30;
      const fibrePct = Math.min(100, Math.round((fibreLogged / fibreGoal) * 100));
      el.gapFibreVal.textContent = `${fibreLogged}g / ${fibreGoal}g avg (${fibrePct}%)`;
      el.progressFibre.style.width = `${fibrePct}%`;

      // Swaps
      el.swapsContainer.innerHTML = "";
      const defaultSwaps = [
        { name: "Foxtail Millet", target_nutrient: "fibre", rationale: "+3.8g Fibre & Lower GI vs White Rice", serving_description: "1 plate cooked" },
        { name: "Double-Toned Milk Tea", target_nutrient: "energy", rationale: "-45 kcal per cup vs Whole Milk", serving_description: "1 standard cup" },
        { name: "Steamed Sprout Chaat", target_nutrient: "protein", rationale: "+9.2g Protein vs Deep Fried Pakoda", serving_description: "1 katori (150g)" },
      ];

      const swaps = (digest.swaps && digest.swaps.length > 0) ? digest.swaps : defaultSwaps;

      swaps.forEach((s) => {
        const card = document.createElement("div");
        card.className = "swap-card";
        card.innerHTML = `
          <div class="swap-header">
            <span class="swap-to">${escapeHtml(s.name || "")}</span>
            <span class="swap-portion">${escapeHtml(s.serving_description || "")}</span>
          </div>
          <div class="swap-benefit">${escapeHtml(s.rationale || s.target_nutrient || "Nutrient dense")}</div>
        `;
        el.swapsContainer.appendChild(card);
      });
    } catch (err) {
      console.warn("Could not load weekly digest:", err);
    }
  }

  // --- Utilities ---
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function showToast(message) {
    let toast = document.getElementById("messmate-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "messmate-toast";
      toast.style.position = "fixed";
      toast.style.bottom = "24px";
      toast.style.left = "50%";
      toast.style.transform = "translateX(-50%)";
      toast.style.background = "#1e293b";
      toast.style.color = "#38bdf8";
      toast.style.padding = "10px 20px";
      toast.style.borderRadius = "9999px";
      toast.style.border = "1px solid rgba(56, 189, 248, 0.4)";
      toast.style.boxShadow = "0 8px 24px rgba(0,0,0,0.5)";
      toast.style.fontSize = "13px";
      toast.style.fontWeight = "600";
      toast.style.zIndex = "9999";
      toast.style.transition = "opacity 0.3s ease";
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.style.opacity = "1";
    setTimeout(() => {
      toast.style.opacity = "0";
    }, 2800);
  }

  // --- App Initialization ---
  async function init() {
    initTabs();
    initContextGate();
    initCapture();
    initOnboarding();
    setupRefillChips([]);

    el.btnQuickAuth.addEventListener("click", quickDemoAuth);

    const isAuthed = await checkAuth();
    if (!isAuthed) {
      // Auto-connect demo account for immediate interactive capability
      quickDemoAuth();
    }
  }

  // Start on DOMContentLoaded
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
