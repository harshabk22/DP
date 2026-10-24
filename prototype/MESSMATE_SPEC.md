# MessMate — Engineering Specification v2.0 (Open-World Consumer Architecture)

**Product:** Downloadable consumer dietary intelligence platform with active tableware calibration
**Audience:** Global consumers (App Store / Google Play), coding agents, and researchers
**Status:** revised v2.0 (Open-World Consumer Architecture). General-purpose downloadable product; not restricted to college mess halls. Multi-context eating (Home vs. Away), universal portion taxonomy, and zero-faking empirical validation.

---

## 0. How to use this document

Read sections 1–3 before writing any code. They contain the invariants that
everything else depends on.

If a task seems to require violating section 2, **stop and ask**. Do not work
around an invariant.

Section 15 is the build order. Do not start a milestone before its predecessor
meets its Definition of Done.

Section 16 lists things we are deliberately **not** building. Do not add them
because they seem useful.

---

## 1. Product summary & philosophy

MessMate is a general-purpose, downloadable mobile dietary tracking application
for anyone seeking effortless, honest nutritional awareness without turning meal
logging into a tedious part-time job.

Traditional dietary tracking tools fail for two reasons:
1. **The Fatigue Trap (MyFitnessPal, Cronometer):** Endless manual searching across
   unverified databases and food-weighing mandates, causing 80% user churn within 30 days.
2. **The Fraudulent Precision Trap (Cal AI, generic AI loggers):** Claiming a single 2D
   photo can magically estimate exact calories, ignoring unobservable 3D bowl depth,
   food density, and invisible absorbed cooking oils (9 kcal/g).

### The MessMate Philosophy: "The Reverse Friction Curve"

> **"Cal AI guesses. MyFitnessPal interrogates. MessMate learns."**

Consumers eat **70–80% of their meals in consistent environments** (home kitchen,
workplace desk, favorite local eateries) using a **finite, recurring set of personal tableware**
(their morning mug, lunch container, dinner plates, and bowls).

MessMate embraces physical reality:
- **Vision is a noisy sensor, not an oracle.** Single-shot 2D mass estimation is an
  ill-posed inverse problem.
- **Active Latent Tableware Learning:** By asking the user for low-friction discrete
  tap counts ($n \in \{1, 2, 3, 4+\}$), the app separates personal tableware volume
  $\theta_{u, m}$ from zero-mean visual observation noise over repeated meals.
- **Attention Budgeting (Max 3 questions/day):** An information-theoretic Value-of-Information
  (VoI) allocator only interrupts the user when variance reduction justifies the cognitive tax.
- **The Reverse Friction Curve:** Unlike legacy trackers where friction is constant,
  MessMate gets *faster and quieter* over time (3 questions/day in Week 1, 1–2 in Week 2,
  $\le 1$ in Week 3). By Day 14, the user's home kitchen is calibrated and meals log in 1 tap.
- **Zero-Faking Science:** Probabilistic conformal prediction intervals $[p_{10}, p_{90}]$
  rather than fake point precision.

### Why this framing (background for contributors)

- Published benchmarks show VLM calorie estimation plateauing around **~80 kcal
  MAE** on Nutrition5k. A 40-model study found model architecture explained
  ~99.6% of performance variance, and that **multi-angle photos and prompt
  engineering gave no significant improvement.** Conclusion: we cannot win by
  prompting harder or capturing more images. The residual error is not
  recoverable from the picture.
- The only remaining lever is **asking the user** — but a 2025 longitudinal study
  (SnappyMeal) found AI follow-up questions became a *new source of friction*.
- Burden destroys retention. In a 12-week intervention, consistent tracking fell
  from **68% of participants in week 1 to 21% by week 12**. MyFitnessPal
  self-monitoring dropped from 5.4 to 1.4 days/week between weeks 4 and 12.
  Users spending >15 min/day logging were **2.1× more likely to abandon** by
  month three.

So: asking is the only way to improve accuracy, and asking is also what kills the
product. That tension is the research problem. Nobody has framed dietary
assessment as explicit accuracy-per-unit-of-burden optimization.

---

## 2. Design invariants

These are non-negotiable. Violating any of them invalidates the research claim,
the product, or our legal position.

| # | Invariant | Why |
|---|---|---|
| **I1** | **The LLM never produces nutrient numbers.** It returns dish labels, household units, counts and gram estimates only. All kcal/protein/fat/carb/micronutrient values come from the database by lookup. | Hallucinated nutrients are unfalsifiable and make the whole system untrustworthy. |
| **I2** | **Every estimate is a distribution, never a point.** Internally we carry `(mean, sd)` or samples. The UI may *display* a single number, but the stored record keeps the spread. | The allocator is driven by variance. If we discard it, there is no project. |
| **I3** | **Max 1 question per meal. Max `budget_total` questions per day.** Default 3. Never exceed, never "just this once". | This is the independent variable of the whole study. |
| **I4** | **Natural multimodal input & tap-only clarification.** The app supports photo capture, natural voice transcripts, and text quick-logging. All assistant clarification questions remain tap-answerable (max 4 options, zero mandatory typing). | Allows rapid logging of unphotographed snacks/drinks without forcing interrogation friction. |
| **I5** | **`corrections` is append-only.** Never UPDATE, never DELETE (except full user erasure). | It is simultaneously the training signal and the research dataset. |
| **I6** | **Consent is per-purpose and unbundled.** Separate toggles for research logging and public dataset inclusion. | Research ethics: consent must be specific, informed, and unambiguous. |
| **I7** | **No account may be created by a user under 18.** | DPDP requires verifiable parental consent for minors. Out of scope for v1. |
| **I8** | **Never display a calorie target below the user's estimated BMR.** Never render shaming or streak-loss messaging. | Eating-disorder safety. See §11. |
| **I9** | **The app must work with zero questions answered.** Questions are an enhancement, never a gate. | A user who ignores every question must still get a usable log. |
| **I10** | **Prompt text and model name are versioned and stored with every inference.** | Results must be reproducible for the paper. |

---

## 3. Glossary

| Term | Meaning |
|---|---|
| **Unit** | A calibrated serving measure: Universal Vessel Archetype (`small_bowl`, `standard_bowl`, `entree_bowl`, `cup_mug`, `dinner_plate`, `tablespoon`), Somatic Hand Ruler (`palm`, `fist`, `thumb`), Modular piece (`slice_piece`), or regional alias (`katori`, `roti`, `serving_spoon`). |
| **Unit scale** (θ) | Volume in $\ln(\text{ml})$ for a vessel or mass in $\ln(\text{g})$ for a piece, *for a specific user*. The latent variable we learn. |
| **Unit count** (n) | How many units are on the plate. Comes from the VLM or the user. |
| **Ask value** (v) | Expected reduction in the meal's total-kcal standard deviation if a given item's portion were resolved exactly. Drives allocation. |
| **Budget** | Questions remaining today for a user (default $\le 3$/day, max 1/meal). |
| **Correction** | Any user-supplied change to a model estimate, whether from a question, refill chip, or manual edit. |
| **Resolution** | Mapping a dish label to a certified row in the federated nutrient database (FNDDS / INDB / IFCT). |

---

## 4. System architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Mobile Client (Flutter iOS / Android + Drift SQLite)       │
│  - "Snap & Shove" capture (<1.5s, 800x800 WebP <120KB)      │
│  - Camera sensor EXIF extraction (focal length, focus dist) │
│  - Zero-permission context gate (foreground BSSID / time-of-│
│    day / 1-tap capture screen chip: [Home 🏠 | Away 🍽️])   │
│  - Delayed notification (+20 min, 2h TTL, in-app fallback)  │
│  - 1-tap interactive notification shade actions (≤4 options)│
│  - Floating refill chips (+1 Slice, +1 Roti, +Half Cup)     │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTPS / JSON
┌──────────────────────────────▼──────────────────────────────┐
│  FastAPI Production Backend (Python 3.11+)                  │
│                                                             │
│  POST /meals ──► local/object storage                       │
│         │                                                   │
│         └──► async task: analyze_meal                       │
│                  │                                          │
│                  ├─ 1. Non-Fiducial Metric Recovery         │
│                  │     (EXIF + Cutlery Anchors + Depth)     │
│                  ├─ 2. VLM Extraction (Gemini Flash-Lite)   │
│                  │     dish_label, vessel_class, count (n)  │
│                  ├─ 3. 3-Tier Federated Database Lookup     │
│                  │     (FNDDS + INDB + IFCT via HNSW vector)│
│                  ├─ 4. Context Gate (Home vs. Away)         │
│                  │     Away: Apply commercial population    │
│                  │     Home: Update personal prior θ_u      │
│                  ├─ 5. Closed-Form VoI Allocator (B ≤ 3)    │
│                  └─ 6. Schedule push question or stay silent│
│                                                             │
│  Daily reset: 02:00 local budget refresh                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  PostgreSQL 15+ with pgvector (staging/prod) / SQLite (dev) │
│  - users, profiles, meals, unit_priors (context-keyed)      │
│  - foods table with 384-dim HNSW vector embedding index     │
└─────────────────────────────────────────────────────────────┘
```

**Stack & Economic Decisions (Consumer Scale):**

- **Mobile Framework:** Flutter (Dart 3.x) with Drift SQLite. Supports both iOS
  and Android. On-device hardware downsampling (800x800 WebP, ~120 KB) guarantees
  sub-second uploads even on congested cellular networks.
- **Unit Economics (<$0.11 USD / User / Month):**
  - $K=1$ deterministic Gemini 2.5 Flash-Lite ($T=0.0$) costs ~$0.075 USD per user
    logging 100 meals/month (~₹6.20 INR).
  - Storage: Cloudflare R2 (zero egress, 30-day purge) costs ~$0.0003 USD/user/mo.
  - Compute & Database: ~$0.010 USD/user/mo.
  - **Total COGS:** ~$0.08–$0.11 USD (~₹6.50–₹9.00 INR) per monthly active user.
  - Enables a high-margin freemium business model (Free Tier + Pro at ₹149/mo or $4.99/mo).
- **Zero-Permission Context Gate:** Avoids invasive background location requests that trigger App Store rejections and user distrust. Combines foreground BSSID hashing when available, time-of-day meal heuristics, and an instant 1-tap chip `[Home 🏠 | Away 🍽️]` on the camera preview.
- **Push Notification Resilience & TTL:** Counteracts aggressive Android OEM battery savers (HyperOS, OxygenOS, OneUI). If the delayed local notification (+20 min) is suppressed or unclicked, an in-app banner renders the pending question upon the user's next app open (TTL = 2 hours, expiring to discard stale recall).
- **Asymmetric Dining Gate:** Protects user priors from restaurant variance by freezing personal Bayesian updates on commercial/dining-out meals.
- **Offline Evaluation vs. Production:**
  - Production uses $K=1$ deterministic Flash-Lite for sub-1.5s latency.
  - Research evaluation harness uses $K=5$ stochastic sampling ($T=0.7$) for empirical variance curves (§13).

---

## 5. Data sources & federated global nutrition engine

To support global consumer downloads (eating Western, Asian, Indian, or Mediterranean diets), the database uses a **3-tier federated resolution engine**:

### 5.1 Tier 1: USDA FNDDS + INDB (Cooked Composite Dishes, ~9,000 foods)
- **USDA FNDDS (Food and Nutrient Database for Dietary Studies):** Covers ~8,000 common cooked composite foods (burgers, pizzas, pastas, salads, stir-fries, bowls).
- **INDB (Indian Nutrient Databank):** Covers ~1,014 cooked Indian home and restaurant dishes.
- **Sub-15ms Resolution:** Indexed locally in PostgreSQL using `pgvector` with a 384-dimensional embedding model (`all-MiniLM-L6-v2`) combined with trigram text matching (`pg_trgm`). Matches 90%+ of consumer meals globally in <15 ms.

### 5.2 Tier 2: Foundation Foods & IFCT 2017 (Raw Ingredients, ~11,000 foods)
- ICMR-NIN IFCT 2017 (528 key foods) and USDA Foundation / SR Legacy tables.
- Used as recipe decomposition fallback when composite dish lookup falls below 0.65 similarity confidence.

### 5.3 Tier 3: Isolated Branded Food Cache (Packaged Goods)
- 350,000+ branded commercial items.
- Segregated from natural language search to eliminate noise; queried strictly when OCR detects brand packaging or when a barcode is scanned.

### 5.4 Nutrition5k (Evaluation Benchmark, Not Shipped in App)
- Google Research (CVPR 2021), CC BY 4.0.
- Used in the offline research harness (§13.1) to validate Bayesian convergence and Pareto allocator curves on 1,000+ overhead food plates with scale-measured masses.

---

## 6. Core algorithms (Open-World Consumer Scale)

### 6.1 Open-world estimation pipeline

```python
analyze_meal(meal_id):
  image     = fetch(meal.image_key)
  exif_meta = extract_exif(meal)                        # focal length, focus distance
  context   = detect_context(user, meal, image)         # Home (E=1) vs Away (E=0)
  metric_s  = estimate_metric_scale(image, exif_meta)   # non-fiducial scale (mm/px)
  
  # Fast VLM extraction (K=1 Gemini Flash-Lite, T=0.0)
  raw_item  = vlm.extract(image, metric_scale=metric_s) # dish_label, vessel_class, count
  items     = resolve_dishes_federated(raw_item)        # FNDDS / INDB / IFCT via vector
  
  for item in items:
      g_mu, g_sd = portion_distribution(user, item, context) # see 6.2
      item.kcal_per_g = item.food.kcal_per_100g / 100
      item.sd_kcal = item.kcal_per_g * g_sd
      
  total = monte_carlo_total(items)                      # see 6.1.3
  question = allocator.maybe_ask(user, meal, items)     # see 6.3
  persist(items, total, question)
```

#### 6.1.1 Non-Fiducial Metric Recovery (3-Tier Optical Hierarchy)
To recover physical scale without requiring users to place coins or calibration cards:
1. **Tier A — Invariant Scene Anchors (Highest Confidence):** VLM identifies standard manufactured scene objects: standard beverage cans ($66\text{ mm}$ diameter), dinner forks ($190\text{ mm}$), soup spoons ($185\text{ mm}$), or standard plate rims ($27.0\text{ cm}$). When present, establishes direct geometric pixel-to-millimeter ratio $s$ [mm/px].
2. **Tier B — Monocular Depth & Sensor EXIF (Relative Volume):** Camera sensor EXIF ($f$, $Z_{\text{focus}}$ where supported by mobile OS) combined with server-side monocular metric depth (Depth Anything v2 Metric, $<45\text{ ms}$) derives rough volumetric relief via closed-form surface integration over the estimated plate plane.
3. **Tier C — Latent Tableware Invariance (The Core Defense):** On devices where EXIF focus distance is quantized/unavailable or in scenes without cutlery (eating with hands at home), **the system does not fail**. The latent tableware prior $\theta_{u, m}$ mathematically absorbs constant instrument and perspective scale biases over repeated meals. Single-shot physical depth errors are treated as zero-mean observation noise ($\tau^2_{\text{eff}}$) filtered out by the Bayesian updater (§6.2).

#### 6.1.2 The "Don't Personalize Restaurants" Asymmetric Dining Gate
A consumer does not own restaurant tableware; commercial portions follow commercial food-service distributions.
- **When Away / Dining Out ($E_t = 0$):** Apply static commercial population distributions. **Freeze the user's prior:** $\Delta \theta_u = 0, \Delta \sigma^2_u = 0$.
- **When Home / Personal Dining ($E_t = 1$):** Update personal tableware prior $\theta_{u, m}$.
- *Result:* Completely prevents restaurant meals from contaminating the user's home tableware posterior with between-context variance.

#### 6.1.3 Monte Carlo Total
```python
S = 2000
totals = np.zeros(S)
for item in items:
    g = np.random.lognormal(item.mu, item.sigma, S)
    totals += g * item.kcal_per_g
return totals.mean(), totals.std(), np.percentile(totals, [10, 90])
```

---

### 6.2 Universal portion taxonomy & Bayesian calibration

#### The Factored Latent Formulation
Portion mass is decoupled into **Vessel Volume** $\times$ **Food Bulk Density**:
$$\ln(\text{mass}_{u, d, m}) = \theta_{u, m} + \ln(\rho_d) + \ln(n)$$
- $\theta_{u, m}$: Latent volume of vessel $m$ owned by user $u$ in $\ln(\text{ml})$.
- $\rho_d$: Objective physical density from nutrient database (e.g., salad $= 0.30\text{ g/ml}$, dal $= 1.05\text{ g/ml}$, rice $= 0.85\text{ g/ml}$).
- $n$: Discrete user count ($n \in \{1, 2, 3, 4+\}$).

#### Universal Vessel Archetypes ($\mathcal{M}$) & Somatic Hand Rulers

| Category | Archetype / Unit | Default Volume | Bulk Density ($\rho$) | Note |
|---|---|---|---|---|
| **Vessels** | `small_bowl` | 150 ml | Dish-dependent | Katori, side salad bowl, dessert cup |
| | `standard_bowl` | 350 ml | Dish-dependent | Cereal bowl, soup bowl, rice bowl |
| | `entree_bowl` | 650 ml | Dish-dependent | Pasta bowl, ramen bowl, grain bowl |
| | `cup_mug` | 240 ml | 1.00 g/ml | Coffee, tea, milk, yogurt |
| | `dinner_plate` | 500 ml mound | Dish-dependent | Main course plate |
| | `tablespoon` | 15 ml | 0.92 g/ml (oils) | Dressing, added fat, condiments |
| **Somatic Rulers** | `palm` | ~85 g (3 oz) | — | Dense proteins (chicken, fish, paneer) |
| | `fist` | ~200 g (1 cup)| — | Dense carbs (rice, potatoes, pasta) |
| | `thumb` | ~15 g (1 tbsp)| — | Fats, butter, oils |
| **Modular** | `slice / piece` | Database std | — | Bread, pizza slice, burger, egg, sushi |

#### Food Density Reference Taxonomy ($\rho_d$)
To prevent subjective density guesses across cuisines, foods map to certified physical bulk densities:

| Food Category | Examples | Density ($\rho$) |
|---|---|---|
| **Liquids, Soups & Dal** | Clear broths, dal, sambar, yogurt, milk, latte | $1.00 - 1.05\text{ g/ml}$ |
| **Cooked Grains & Tubers** | Steamed rice, quinoa, pasta, mashed potato | $0.80 - 0.90\text{ g/ml}$ |
| **Dense Proteins** | Cooked chicken, fish fillet, paneer cubes, tofu | $1.05 - 1.15\text{ g/ml}$ |
| **Loose Greens & Salads** | Raw spinach, caesar salad, microgreens | $0.25 - 0.35\text{ g/ml}$ |
| **Snacks & Puffed Foods** | Popcorn, potato crisps, papad, kurkure | $0.15 - 0.25\text{ g/ml}$ |
| **Fats & Oils** | Olive oil, ghee, melted butter, dressings | $0.92\text{ g/ml}$ |

#### Day 1 "Visual Kitchen Mirror" Onboarding (Zero Coin Hassle)
During initial setup (20 seconds), the app displays 3 silhouette sizes for standard bowls and plates. The user taps the closest match, immediately seeding $\mu_0$ with an informative prior ($\sigma^2_0 = 0.05$) without taking reference photos.

#### Update Rule (Discretization & Context-Gated Conjugacy)
```python
def update_unit_prior(prior_mu, prior_s2, g_vlm, n_user, rho_dish, is_home, consecutive_rejects=0):
    # Rule 1: Never update personal prior on commercial / dining-out meals
    if not is_home or n_user is None or n_user < 0.5:
        return prior_mu, prior_s2, consecutive_rejects

    # Protect against non-positive inputs
    g_eff = max(float(g_vlm), 1.0)
    rho_eff = max(float(rho_dish), 0.1)
    v_obs = g_eff / rho_eff
    x = math.log(v_obs / float(n_user))

    # Observation variance + tap discretization error: Var(delta) = 1/12 => 1 / (12*n^2) in log-space
    tau2_eff = max(0.04, 0.04) + 1.0 / (12.0 * (float(n_user) ** 2))

    # Dynamic 2.8-sigma residual gate (rejects vessel misclassifications)
    sigma_pred = math.sqrt(prior_s2 + tau2_eff)
    residual = abs(x - prior_mu)
    
    if residual > 2.8 * sigma_pred:
        consecutive_rejects += 1
        # 3-Strike Outlier Reset: if 3 consecutive meals deviate in the same direction,
        # user likely changed tableware set; inflate variance to adapt rather than freeze.
        if consecutive_rejects >= 3:
            prior_s2 = max(prior_s2, 0.15)
            consecutive_rejects = 0
        return prior_mu, prior_s2, consecutive_rejects

    # Successful update: reset consecutive rejects
    consecutive_rejects = 0

    # Conjugate Normal-Normal update
    prec_prior = 1.0 / prior_s2
    prec_obs   = 1.0 / tau2_eff
    s2_new = 1.0 / (prec_prior + prec_obs)
    mu_new = s2_new * (prior_mu * prec_prior + x * prec_obs)

    # Impose variance floor (prevent false mathematical certainty)
    s2_new = max(s2_new, 0.01)
    return mu_new, s2_new, consecutive_rejects
```

#### Inference

Given unit `m` and count `n`:

```
ln(grams) ~ Normal(mu_log + ln(n), sigma2_log + meal_noise)
grams_mean = exp(mu + sigma2/2)
grams_sd   = grams_mean * sqrt(exp(sigma2) - 1)
```

Add `meal_noise = 0.02` for genuine serving-to-serving variation.

---

### 6.3 Budget allocator

#### Ask value (Thali-Robust Formulation)

To eliminate the Pythagorean suppression where multi-dish thalis silences questions,
we evaluate the item's direct calorie uncertainty:

```
v_i = sd_i = kcal_per_g_i * grams_sd_i
```

Units: kcal. Robust across 1-dish snacks and 5-dish full thalis.

#### Online allocation (Budget-Pressure Policy)

```python
def maybe_ask(user, meal, items):
    B = budget_remaining(user, today)
    if B <= 0:
        return None

    candidates = [i for i in items if i.askable]
    if not candidates:
        return None

    best = max(candidates, key=lambda i: i.sd_kcal)

    if best.sd_kcal < 25.0:            # V_FLOOR = 25 kcal
        return None                   # never interrupt for trivial gains

    meals_remaining = max(expected_remaining_meals_today(user), 1)
    pressure = B / meals_remaining    # e.g., 3 budget / 3 meals = 1.0

    # Cold-start fix: if budget pressure >= 1.0, ask top item directly
    if pressure >= 1.0:
        return build_question(best)

    # When budget is tight (P < 1.0), compare against user's history
    if len(user.v_history) < 15:
        threshold = 35.0              # sensible early threshold
    else:
        threshold = user.v_history.quantile(1.0 - pressure)

    if best.sd_kcal >= threshold:
        return build_question(best)
    return None
```

- Budget resets at 02:00 local time.
- **Atomic reservation:** Protected via DB row-level locks (`FOR UPDATE`).

#### Question types

| Type | Prompt | Options | Typical value |
|---|---|---|---|
| `unit_count` | "How many rotis?" | 1 / 2 / 3 / 4+ | Highest yield |
| `dish_disambiguation` | "Rajma or Chana?" | ≤4 labels | Used on low VLM confidence |

All tap-only (**I4**), delivered via delayed local push notification (+20 min).

#### Policies to implement (for evaluation)

- `none` — never ask (lower bound)
- `always` — ask about every item with $v > 0$ (upper bound)
- `random` — random item, same budget B
- `threshold` — fixed $\theta$, no budget
- `allocator` — ours (Budget-Pressure VoI)

---

## 7. Database schema

PostgreSQL. Use Alembic migrations. `gen_random_uuid()` requires `pgcrypto`;
fuzzy matching requires `pg_trgm`.

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- fuzzy dish matching
CREATE EXTENSION IF NOT EXISTS citext;     -- case-insensitive email

-- ============ identity ============
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           CITEXT UNIQUE NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at      TIMESTAMPTZ,
    study_arm       TEXT CHECK (study_arm IN ('A_fixed','B_adaptive'))
);

CREATE TABLE profiles (
    user_id         UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    -- NOTE: the under-18 check (I7) CANNOT be a CHECK constraint — Postgres
    -- rejects non-IMMUTABLE functions like now() inside CHECK. Enforce it in
    -- the API layer at registration AND at profile update, with a test.
    birth_year      INT  NOT NULL CHECK (birth_year BETWEEN 1900 AND 2100),
    sex             TEXT NOT NULL CHECK (sex IN ('male','female','other')),
    height_cm       NUMERIC(5,1),
    weight_kg       NUMERIC(5,1),
    activity_level  TEXT CHECK (activity_level IN
                      ('sedentary','light','moderate','active','very_active')),
    goal            TEXT CHECK (goal IN ('maintain','lose','gain','awareness')),
    diet_type       TEXT CHECK (diet_type IN
                      ('veg','non_veg','eggetarian','vegan','jain')),
    allergies       TEXT[] DEFAULT '{}',
    conditions      TEXT[] DEFAULT '{}',
    region          TEXT,
    locale          TEXT DEFAULT 'en-IN',
    cafeteria_id    TEXT, -- optional campus/corporate cafeteria tag; null for general consumers
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- I6: one row PER PURPOSE. Never bundle.
CREATE TABLE consents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    purpose         TEXT NOT NULL CHECK (purpose IN
                      ('core_logging','model_improvement','research_dataset')),
    granted         BOOLEAN NOT NULL,
    notice_version  TEXT NOT NULL,
    granted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at      TIMESTAMPTZ
);
CREATE INDEX ON consents (user_id, purpose, granted_at DESC);

-- ============ nutrient reference ============
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector for semantic dish matching

CREATE TABLE foods (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL CHECK (source IN ('FNDDS','INDB','IFCT2017','CURATED')),
    source_code     TEXT,
    name_en         TEXT NOT NULL,
    name_local      JSONB DEFAULT '{}',
    aliases         TEXT[] DEFAULT '{}',
    food_group      TEXT,
    density_g_per_ml NUMERIC(4,2) DEFAULT 1.00,
    kcal_per_100g   NUMERIC(7,2) NOT NULL,
    protein_g       NUMERIC(6,2),
    fat_g           NUMERIC(6,2),
    carb_g          NUMERIC(6,2),
    fibre_g         NUMERIC(6,2),
    micros          JSONB DEFAULT '{}',
    default_unit    TEXT,
    pop_grams_per_unit NUMERIC(7,2),
    embedding       vector(384),
    provenance_note TEXT,
    UNIQUE (source, source_code)
);
CREATE INDEX foods_name_trgm ON foods USING gin (name_en gin_trgm_ops);
CREATE INDEX foods_embedding_idx ON foods USING hnsw (embedding vector_cosine_ops);

-- Optional institutional menu (for campus mess or corporate cafeterias; null for home/restaurant eaters)
CREATE TABLE cafeteria_menus (
    id              BIGSERIAL PRIMARY KEY,
    cafeteria_id    TEXT NOT NULL DEFAULT 'campus_default',
    day_of_week     INT CHECK (day_of_week BETWEEN 0 AND 6),
    meal_slot       TEXT CHECK (meal_slot IN ('breakfast','lunch','snack','dinner')),
    food_id         BIGINT REFERENCES foods(id),
    standard_serving_g NUMERIC(7,2)
);

-- ============ logging ============
CREATE TABLE meals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    captured_at     TIMESTAMPTZ NOT NULL,
    local_date      DATE NOT NULL,
    meal_slot       TEXT,
    image_key       TEXT,
    image_purge_at  TIMESTAMPTZ,
    source          TEXT NOT NULL CHECK (source IN ('photo','voice','manual')),
    status          TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending','analyzed','failed','discarded')),
    kcal_mean       NUMERIC(8,2),
    kcal_sd         NUMERIC(8,2),
    kcal_p10        NUMERIC(8,2),
    kcal_p90        NUMERIC(8,2),
    latency_ms      INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON meals (user_id, local_date);

CREATE TABLE meal_items (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meal_id         UUID NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    food_id         BIGINT REFERENCES foods(id),
    dish_label_raw  TEXT NOT NULL,
    resolution_tier TEXT CHECK (resolution_tier IN
                      ('exact','fuzzy','decomposed','unresolved')),
    dish_uncertain  BOOLEAN NOT NULL DEFAULT false,
    unit            TEXT,
    unit_count      NUMERIC(5,2),
    grams_mean      NUMERIC(8,2),
    grams_sd        NUMERIC(8,2),
    kcal_mean       NUMERIC(8,2),
    kcal_sd         NUMERIC(8,2),
    ask_value       NUMERIC(8,2),
    portion_source  TEXT CHECK (portion_source IN ('model','prior','user')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meal_id         UUID NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    model_name      TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,          -- I10
    k_samples       INT NOT NULL,
    temperature     NUMERIC(3,2),
    raw_responses   JSONB NOT NULL,
    latency_ms      INT,
    cost_usd        NUMERIC(10,6),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============ the learning loop ============
CREATE TABLE questions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meal_id         UUID NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    meal_item_id    UUID REFERENCES meal_items(id) ON DELETE CASCADE,
    question_type   TEXT NOT NULL CHECK (question_type IN
                      ('unit_count','unit_size','oil_level','dish_disambiguation')),
    prompt_text     TEXT NOT NULL,
    options         JSONB NOT NULL,
    ask_value       NUMERIC(8,2) NOT NULL,
    threshold_used  NUMERIC(8,2),
    budget_remaining INT NOT NULL,
    policy_name     TEXT NOT NULL,
    asked_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at     TIMESTAMPTZ,
    answer          JSONB,
    skipped         BOOLEAN NOT NULL DEFAULT false
);

-- I5: APPEND ONLY. No UPDATE, no DELETE.
-- (Enforce with a BEFORE UPDATE OR DELETE trigger; permit deletion only during
-- full user erasure when session setting 'messmate.hard_delete = on' is active).
CREATE TABLE corrections (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    meal_item_id    UUID REFERENCES meal_items(id) ON DELETE CASCADE,
    field           TEXT NOT NULL CHECK (field IN
                      ('dish','unit','unit_count','grams','oil_level')),
    old_value       JSONB,
    new_value       JSONB NOT NULL,
    source          TEXT NOT NULL CHECK (source IN
                      ('question','manual_edit','scale')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON corrections (user_id, created_at);

CREATE TABLE unit_priors (
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    unit            TEXT NOT NULL,
    context         TEXT NOT NULL DEFAULT 'personal_home'
                      CHECK (context IN ('personal_home','personal_office','commercial')),
    mu_log          NUMERIC(8,5) NOT NULL,
    sigma2_log      NUMERIC(8,5) NOT NULL,
    n_obs           INT NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, unit, context)
);

-- full history of posterior evolution — this IS Figure 2
CREATE TABLE unit_prior_history (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    unit            TEXT NOT NULL,
    context         TEXT NOT NULL DEFAULT 'personal_home',
    mu_log          NUMERIC(8,5) NOT NULL,
    sigma2_log      NUMERIC(8,5) NOT NULL,
    n_obs           INT NOT NULL,
    triggered_by    UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE daily_budgets (
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    local_date      DATE NOT NULL,
    budget_total    INT NOT NULL DEFAULT 3,
    budget_used     INT NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, local_date)
);

-- ============ engagement & telemetry ============
CREATE TABLE events (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    type            TEXT NOT NULL,
    meta            JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON events (user_id, type, created_at);
```

**Events to emit** (needed for research analysis): `capture_submitted`,
`analysis_shown`, `question_shown`, `question_answered`, `question_skipped`,
`manual_edit`, `refill_added`.

---

## 8. API contract

Base: `/v1`. Bearer token auth. All responses JSON.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/auth/register` | email + password; enforces **I7** |
| `POST` | `/auth/login` | |
| `GET`/`PUT` | `/me/profile` | |
| `GET`/`POST` | `/me/consents` | privacy consent settings; **I6** |
| `DELETE` | `/me` | full account erasure; **must actually delete** |
| `GET` | `/me/export` | data portability, JSON bundle |
| `POST` | `/meals/photo` | multipart image → instant analysis result (<1.0s) |
| `POST` | `/meals/text` | natural text or voice transcript quick-log |
| `GET` | `/meals/{id}` | meal details and breakdown |
| `POST` | `/meals/{id}/answer` | 1-tap clarification chip response |
| `PATCH` | `/meal-items/{id}` | manual correction or refill chip (`+1 Roti`) |
| `DELETE` | `/meals/{id}` | discard a meal |
| `POST` | `/chat` | conversational nutrition co-pilot Q&A |
| `GET` | `/days/{date}` | daily rollup, macro totals & 19:00 dinner swap advice |
| `GET` | `/weeks/{iso_week}` | weekly summary |

**`GET /meals/{id}` response shape:**

```json
{
  "meal_id": "uuid",
  "status": "analyzed",
  "captured_at": "2026-09-21T13:04:00+05:30",
  "total": { "kcal_mean": 612.4, "kcal_sd": 88.1,
             "kcal_p10": 502.0, "kcal_p90": 728.0,
             "protein_g": 18.2, "fat_g": 21.4, "carb_g": 84.6 },
  "items": [
    { "id": "uuid", "dish_label": "sambar",
      "food_id": 4412, "resolution_tier": "exact",
      "unit": "katori", "unit_count": 1,
      "grams_mean": 168.0, "grams_sd": 31.2,
      "kcal_mean": 142.8, "kcal_sd": 26.5,
      "portion_source": "prior", "ask_value": 26.5 }
  ],
  "question": {
    "id": "uuid", "type": "unit_count",
    "prompt": "How many rotis?",
    "options": [{"label":"1","value":1},{"label":"2","value":2},
                {"label":"3","value":3},{"label":"4+","value":4}]
  },
  "budget": { "total": 3, "used": 1 }
}
```

`question` is `null` when the allocator stays silent. The client must render a
complete, usable result in that case (**I9**).

---

## 9. VLM contract

### 9.1 Rules

- **Dual-Mode Execution:**
  - **Production ($K=1$):** Single deterministic call using Gemini 2.5 Flash-Lite (`temperature = 0.0`) for sub-1.5s latency and unit economics ($\le \$0.075$/user/month).
  - **Offline Research Evaluation ($K=5$):** 5 concurrent stochastic samples (`temperature = 0.7`) to extract empirical perceptual variance curves for scientific validation (§13.2).
- Structured JSON output enforced by strict Pydantic schema. Reject and retry once on parse failure; after two failures mark the meal `failed`.
- **The prompt must not ask for calories, macros, or any nutrient value (I1).**
- Store `prompt_version` and `model_name` with every run (**I10**). Bump version on any prompt edit.
- Timeout: 15s in production ($K=1$), 20s per sample in research ($K=5$).

### 9.2 Output schema

```json
{
  "plate_type": "thali | single_bowl | single_plate | meal_prep_box | packaged | unclear",
  "items": [
    {
      "dish_label": "string, lowercase, common culinary name (e.g., 'grilled salmon', 'dal makhani', 'caesar salad')",
      "ingredient_guess": ["string"],
      "vessel_archetype": "small_bowl | standard_bowl | entree_bowl | cup_mug | dinner_plate | tablespoon | palm | fist | thumb | slice_piece | none",
      "unit_count": 1.5,
      "grams_estimate": 180.0,
      "occlusion": "none | partial | heavy",
      "confidence": 0.88
    }
  ],
  "scene_anchors_detected": ["cutlery_fork", "beverage_can", "plate_rim", "none"],
  "notes": "string, max 200 chars"
}
```

### 9.3 Prompt skeleton (v1)

```
You are an expert food perception model for an open-world dietary tracking system.

Return ONLY JSON matching the provided schema. No prose or explanations.

For each distinct food item visible on the plate or vessel:
- dish_label: common culinary name in lowercase (e.g., "sambar", "steamed jasmine rice", "grilled chicken breast", "penne arrabbiata"). Use common local/transliterated culinary names where appropriate.
- vessel_archetype: identify the serving vessel archetype or hand ruler from:
  ["small_bowl", "standard_bowl", "entree_bowl", "cup_mug", "dinner_plate", "tablespoon", "palm", "fist", "thumb", "slice_piece", "none"].
- unit_count: number of units or fraction (e.g., 1, 2, 0.5, 1.5). Set null if uncountable.
- grams_estimate: your best physical estimate of edible mass in grams based on visual volume.
- occlusion: visual visibility ("none", "partial", "heavy").
- confidence: float between 0.0 and 1.0 representing identification certainty.

Hard rules:
- NEVER estimate calories, macronutrients, or nutritional values (Invariant I1).
- Never merge distinct dishes into a single entry.
- Identify any standard manufactured objects for non-fiducial scene scaling (forks, spoons, standard cans, plate rims).
- If uncertain between two candidate dishes, select the more likely dish and flag confidence < 0.60 for potential user disambiguation.
```

Keep a frozen copy at `prompts/extract_v1.txt`. **Never edit in place** — create
`extract_v2.txt` and bump the version.

---

## 10. In-app weekly digest & menu swaps

Weekly summary rendered directly inside the Flutter app every Sunday morning.
No email server or inbound webhook required.

**Contents:**
1. Days logged this week (no streak/loss framing — **I8**).
2. Two nutrient gaps vs. ICMR-NIN recommended intakes (protein, fibre).
3. Three concrete food swaps drawn from curated database foods (or active cafeteria menu if linked).
4. Clean visualization of personal calorie spread.

The recommendation engine is **rule-based**, not generative:

```python
for nutrient in ['protein', 'fibre']:
    gap = recommended_weekly[nutrient] - consumed_weekly[nutrient]
    if gap > 0.15 * recommended_weekly[nutrient]:
        # For general consumers, pool from curated high-density foods.
        # For campus / workplace users with a cafeteria feed, pool from active cafeteria menu.
        candidates = get_swap_candidates(user.diet_type, user.cafeteria_id)
        rank_by(candidates, key=lambda d: d[nutrient] / d.kcal)
        emit_top_3(candidates)
```

No LLM in this path. Deterministic and auditable.

---

## 11. Safety guardrails

Implemented in the core business logic layer with unit tests.

| Rule | Implementation |
|---|---|
| No target below BMR | Compute BMR (Mifflin–St Jeor). Clamp any displayed target to `max(target, BMR)`. |
| No rapid-loss goals | Reject weight-loss rates > 1% body weight/week at profile save. |
| No shaming | Ban "over budget", "you failed", streak-loss language from all copy. Review every string. |
| Non-clinical disclaimer | Persistent footer: Not medical advice; academic research prototype. |
| Under-18 | Blocked at registration (**I7**). |

---

## 12. Privacy & research consent

Streamlined for university academic study ethics:

- [ ] Clear participant information sheet shown before registration.
- [ ] Explicit study consent toggle for research logging and data inclusion (**I6**).
- [ ] `DELETE /me` performs actual deletion: DB rows, cached images.
- [ ] `GET /me/export` returns complete JSON bundle.
- [ ] All exported evaluation datasets are strictly de-identified (user IDs replaced
      with anonymous hashes `u_01`, `u_02`).

---

## 13. The "zero-faking" scientific evaluation protocol

Lives in `eval/`. Runs directly from the CLI to produce paper figures, Pareto curves, and validation statistics.

To submit to top peer-reviewed digital health and medical informatics venues (*IEEE BHI*, *ACM CHIL*, *JMIR*, *Nature Digital Medicine*) without making fraudulent or unverifiable claims, we establish a **three-tier evaluation protocol**:

### 13.1 Tier 1 — Controlled Multi-Cuisine Weighed Benchmark ($N = 200$ dishes)
- **Dataset:** 200 multi-ingredient dishes across Western, Asian, Indian, and Mediterranean cuisines, with components weighed individually to $\pm 1$g precision using a digital kitchen scale.
- **Hypothesis:** Decoupling visual perception from certified food composition tables (Invariant I1) eliminates catastrophic nutrient hallucinations produced by zero-shot VLM wrappers like Cal AI.
- **Metric:** Macro/Kcal Mean Absolute Error (MAE) and Mean Percentage Error (MPE) against ground truth.

### 13.2 Tier 2 — Agent-Based Longitudinal Simulation on Nutrition5k ($N = 200$ personas, 60 days)
- Uses the 1,000+ real overhead images from Google Research's Nutrition5k benchmark.
- Simulates 200 consumer personas with distinct tableware priors drawn from broad population log-normal distributions.
- Models user answering behavior and re-runs meal histories under 5 allocator policies (`none`, `random`, `static_threshold`, `oracle_always`, `messmate_voi`).
- **Paper Deliverables:**
  - **Figure 1 (Pareto Frontier):** Kcal MAE vs. Questions Asked. Proves MessMate achieves 85% of Oracle accuracy at $\le 0.8$ questions/meal, outperforming random/greedy baselines.
  - **Figure 2a (Parameter Convergence):** Posterior $\mu_{\log} \to \theta^*$ convergence curves over meal index (converging within 12–18 meals).

### 13.3 Tier 3 — Opt-in Verified Consumer Scale Cohort ($N = 15\text{--}20$ households, 14 days)
- **Protocol:** 15–20 real consumer participants in their personal homes are provided digital kitchen scales to log parallel Weighed Food Records (WFR) for 14 days.
- **Zero-Faking Clinical Standards:**
  - **Bland-Altman 95% Limits of Agreement (LoA):** Difference vs. mean plot comparing uncalibrated Week 1 vs. calibrated Week 2. Evaluates whether the 95% LoA narrows significantly as tableware posteriors sharpen.
  - **Lin’s Concordance Correlation Coefficient (CCC):** Measures combined precision and accuracy against the $45^\circ$ identity line.
  - **Calibrated Split-Conformal Prediction Intervals:** Nominal Monte Carlo draws yield Bayesian credible bounds $[p_{10}, p_{90}]_{\text{nominal}}$. To provide distribution-free finite-sample guarantees against unmodeled culinary variances (oil, water content), we scale bounds by the empirical non-conformity factor $\hat{q}_{0.90}$ fitted on calibration split $D_{\text{cal}}$:
    $$R_i = \frac{|Y_{i,\text{true}} - \hat{Y}_i|}{\hat{\sigma}_i}, \quad \hat{q}_{0.90} = \text{Quantile}\left(\{R_i\}_{i \in D_{\text{cal}}}, \frac{\lceil(N+1)(0.90)\rceil}{N}\right)$$
    Yielding conformal intervals $[p_{10}, p_{90}]_{\text{calibrated}} = [\hat{Y} - \hat{q}_{0.90}\hat{\sigma}, \hat{Y} + \hat{q}_{0.90}\hat{\sigma}]$ rigorously satisfying:
    $$P(Y_{\text{true}} \in [p_{10}, p_{90}]) \ge 0.90$$

---

## 14. Repo layout

```
messmate/
├── README.md
├── docs/
│   ├── SPEC.md              ← this file
│   ├── decisions/           ← ADRs, one per non-obvious choice
│   └── incident.md
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/v1/          ← routers, one per resource
│   │   ├── core/            ← config, auth, safety middleware
│   │   ├── models/          ← SQLAlchemy / pgvector
│   │   ├── schemas/         ← Pydantic v2
│   │   ├── services/
│   │   │   ├── vlm/         ← VLMClient (Gemini Flash-Lite)
│   │   │   ├── resolution/  ← 3-tier federated HNSW vector resolution
│   │   │   ├── priors/      ← context-gated Bayesian update (§6.2)
│   │   │   ├── allocator/   ← budget-pressure VoI allocator (§6.3)
│   │   │   ├── nutrition/   ← Monte Carlo composition
│   │   │   └── digest/      ← weekly in-app summary & swaps
│   │   └── tasks/           ← async background tasks
│   ├── alembic/
│   └── tests/
├── mobile/                  ← Flutter client (iOS & Android, Drift SQLite)
├── scripts/ingest/          ← USDA FNDDS, INDB, IFCT 2017 migrations
├── prompts/                 ← extract_v1.txt, frozen
├── eval/
│   ├── sim/                 ← Tier 2 (Nutrition5k simulation)
│   ├── benchmark/           ← Tier 1 (controlled 200 weighed dishes)
│   ├── cohort/              ← Tier 3 (Bland-Altman consumer analysis)
│   └── figures/             ← generated paper figures (Figure 1, 2, Table 1)
└── docker-compose.yml
```

---

## 15. Build order (16-Week Consumer & Research Plan)

Calibrated for a standard 16-week semester or product development cycle.

### Phase 1 — Foundations & Federated Database (Weeks 1–3)
- Ingest USDA FNDDS (~8,000 cooked foods) and INDB/IFCT into PostgreSQL with `pgvector` HNSW embedding index.
- Setup FastAPI backend with Pydantic v2 schemas and non-fiducial EXIF extraction.
- **DoD:** Sub-15ms semantic matching on open-world foods (e.g., "grilled salmon with quinoa", "rajma chawal").

### Phase 2 — Multi-Cuisine Ground-Truth Dataset Sprint (Weeks 4–6)
- Collect and weigh Tier 1 benchmark (200 multi-cuisine dishes with $\pm 1$g digital kitchen scale).
- Partition into calibration and evaluation sets; tag canonical vessel archetypes.
- **DoD:** Clean 200-dish open benchmark released with scale ground truth.

### Phase 3 — Algorithm Engine & Nutrition5k Simulation (Weeks 7–9)
- Implement context-gated Bayesian updater (Home vs. Away) with discretization noise compensation.
- Implement thali-robust ask-value ($v_i = sd_i$) and budget-pressure allocation policy.
- Run Tier 2 simulation on 1,000 Nutrition5k plates across 200 simulated personas.
- **DoD:** **Figure 1 (Pareto Frontier) and Figure 2a (Parameter Convergence) successfully generated.**

### Phase 4 — Flutter Consumer App & Micro-Cohort (Weeks 10–12)
- Build cross-platform Flutter client with on-device WebP compression and local Drift SQLite.
- Implement Day 1 "Visual Kitchen Mirror" onboarding (select 3 bowl silhouettes in 20s).
- Implement local push notification scheduled at $T + 20$ minutes with 1-tap interactive action buttons.
- Run 14-day Tier 3 micro-cohort with 15 consumer households using kitchen scales.
- **DoD:** Complete 14-day parallel WFR dataset for Bland-Altman LoA evaluation.

### Phase 5 — Scientific Analysis, Paper Writing & App Store Release (Weeks 13–16)
- Compute Bland-Altman 95% LoA and Lin's CCC; format Table 1 and paper figures.
- Finalize 6–8 page paper for submission to IEEE BHI / ACM CHIL / JMIR.
- Package and deploy consumer app to App Store and Google Play internal testing.
- **DoD:** Paper submitted; live, working consumer app ready for public download and final-year evaluation defense.

---

## 16. Non-goals

Do **not** build these. If one seems necessary, raise it — don't just add it.

- ❌ **A custom CNN / fine-tuned vision model.** Hosted Gemini Flash-Lite dominates on open-world classification.
- ❌ **Fiducial markers / coins / credit cards on the plate.** Destroys consumer UX. We use sensor EXIF + scene cutlery anchors + server metric depth.
- ❌ **Personalizing restaurant meals.** Violates the "Don't Personalize Restaurants" Theorem; commercial meals freeze user priors.
- ❌ **Fake point-estimate calories.** We reject "642 kcal" illusions and render calibrated honest ranges $[p_{10}, p_{90}]$.
- ❌ **Runtime K=5 VLM ensembling.** Destroys unit economics and increases latency. $K=1$ deterministic in prod ($<1.0\text{s}$), $K=5$ in offline research.
- ❌ **Unsolicited pop-quiz proctoring.** The assistant never badgers the user with arbitrary pop quizzes. Clarifications are quiet, optional micro-actions.
- ❌ **Generative nutrient hallucinations.** The conversational co-pilot derives all calorie and macro numbers strictly by joining certified database rows (Invariant I1), never free-wheeling LLM generation.
- ❌ **Barcode scanning as the primary flow.** Supported only as Tier 3 fallback for packaged goods.

---

## 17. Settled decisions

1. **Product Scope:** Open-world downloadable consumer application (iOS & Android) with the Reverse Friction Curve.
2. **Context Partitioning:** Home dining updates personal tableware priors $\theta_{u, m}$; restaurant/away dining uses commercial population distributions and freezes personal priors.
3. **Nutrition Database:** Federated global engine: Tier 1 USDA FNDDS + INDB (cooked composite dishes, HNSW vector index <15ms), Tier 2 Foundation/IFCT raw fallback, Tier 3 Branded isolation.
4. **Validation Protocol:** Three-tier Zero-Faking evaluation: Tier 1 (200 weighed multi-cuisine dishes), Tier 2 (Nutrition5k 200-persona simulation), Tier 3 (15-household 14-day WFR cohort for Bland-Altman LoA).
5. **Mobile Client:** Flutter with Drift SQLite; <1.5s "Snap & Shove" capture, local delayed push notification (+20 min), and persistent refill chips.

---

## 18. References

- Nutrition5k (CVPR 2021) — `github.com/google-research-datasets/Nutrition5k`,
  CC BY 4.0. Per-ingredient mass in `metadata/dish_metadata_cafe{1,2}.csv`.
- IFCT 2017, ICMR-NIN — `nin.res.in/ebooks/IFCT2017.pdf`;
  machine-readable at `github.com/ifct2017` / npm `ifct2017`.
- INDB — *Development of an Indian Food Composition Database*, PMC11277795.
  1,095 items + 1,014 recipes.
- VLM dietary assessment benchmark (2026) — ~80.7 kcal MAE, CCC 0.767 on
  Nutrition5k; ingredient Jaccard ~0.655.
- *Model architecture dominates nutritional estimation accuracy in vision-language
  systems* — Scientific Reports, 2026. Architecture ≈99.6% of variance;
  multi-angle and prompt engineering not significant.
- SnappyMeal (arXiv 2511.03907) — multimodal AI food logging; AI follow-ups
  improved clarity but also created friction.
- Adherence: 68% → 21% over 12 weeks; MyFitnessPal 5.4 → 1.4 days/week weeks
  4→12; >15 min/day logging → 2.1× abandonment by month 3.
- Khana (arXiv 2509.06006) — ~131K Indian food images, 80 labels. No mass labels.
- DPDP Act 2023 + DPDP Rules 2025 (notified Nov 2025; phased to May 2027).

---

*End of spec. Update this document before changing behaviour it describes.*
