# AI-Based Smart Meal Analyzer and Personalized Nutrition Recommendation System

## Literature Review of Verified Papers Published from 2022 to 2026

**Review date:** 23 September 2026  
**Project scope:** B.Tech design project on image-based meal analysis, calorie/nutrient estimation, food logging, and personalized nutrition recommendations.

---

## 1. Review question

Which recently published research best supports the design of an AI-based Smart Meal Analyzer and Personalized Nutrition Recommendation System, and what can be transferred from each paper into a defensible project architecture?

The review covers five connected functions:

1. Food and ingredient recognition.
2. Food segmentation and portion/volume estimation.
3. Calorie, macro- and micronutrient estimation.
4. Image-assisted food logging and intake validation.
5. Personalized, health-aware nutrition and meal recommendations.

---

## 2. Search and verification protocol

### 2.1 Sources used

The candidate pool and final papers were checked across:

- OpenAlex works search, DOI resolution, citation metadata, and retraction flags.
- Publisher records, including IEEE Xplore, ACM Digital Library, Nature, Springer, Elsevier, Frontiers, JMIR, Oxford Academic, and MDPI.
- PubMed, PubMed Central, and Europe PMC.
- Legal publisher PDFs, CVF accepted versions, and the NSF Public Access Repository copy when the publisher PDF was blocked.
- Built-in web search for publisher metadata, legal full-text availability, and independent corroborating records.

### 2.2 Search concepts

The main searches combined terms for:

- `food image recognition`, `meal image`, `food segmentation`, `ingredient recognition`;
- `calorie estimation`, `nutrient estimation`, `portion estimation`, `volume estimation`;
- `image-based dietary assessment`, `food logging`, `AI mobile diet app`;
- `personalized nutrition`, `meal recommendation`, `health-aware recommender`, `nutrition recommendation`;
- `multimodal LLM`, `vision-language model`, `ChatGPT`, `reinforcement learning`, `food recommender`.

Representative OpenAlex searches returned 117 candidate works for image-based calorie/nutrient estimation, 226 for AI-personalized nutrition, and 256 for image-assisted dietary assessment/food logging before screening.

### 2.3 Inclusion criteria

A paper was included only when it satisfied all of the following:

- Final peer-reviewed journal or conference publication.
- Publication year from 2022 through 2026; 2026 means published by the review date.
- Direct relevance to at least one core project function.
- Verifiable title, authors, venue, publication status, and DOI.
- No retraction, withdrawal, expression of concern, or suspicious publisher record.
- Sufficient methodological detail to identify the dataset, model, evaluation, strengths, and limitations.
- Legal full text available from a publisher, CVF, or the NSF Public Access Repository copy.

### 2.4 Exclusion criteria

The following were excluded:

- Theses, dissertations, datasets without peer-reviewed evaluation, and preprint-only records.
- Papers outside 2022–2026.
- Generic AI commentary without meal analysis or nutrition personalization.
- Retracted, withdrawn, archived-only, or untraceable records.
- Papers from predatory or unverifiable venues.
- Classification-only studies with no useful contribution to portion, nutrient, validation, or recommendation design.
- Duplicate workshop/preprint and journal versions counted as separate papers.

### 2.5 Verification outcome

All 20 selected DOI records matched exactly in OpenAlex. None was marked retracted. Publisher or PubMed/Europe PMC metadata agreed with the selected title, venue, and publication year. The final set contains:

- 3 papers from 2022.
- 4 papers from 2023.
- 5 papers from 2024.
- 5 papers from 2025.
- 3 papers published by 23 September 2026.

Nineteen local files are final publisher or CVF accepted-version PDFs. Paper 17 is the publicly accessible NSF Public Access Repository copy of the ACM KDD publication; the local file is ACM-formatted, carries the KDD citation and DOI, and has no arXiv header. No local PDF is sourced from arXiv; every citation is a final peer-reviewed IEEE/CVF, ACM, journal, or conference record verified by DOI and publisher metadata.

---

## 3. Selected papers at a glance

| # | Year | Paper | Main role | Local full text |
|---:|---:|---|---|---|
| 1 | 2022 | Applying Image-Based Food-Recognition Systems on Dietary Assessment | End-to-end systematic review | `papers/01_2022_Dalakleidi_Image_Based_Food_Recognition_Review.pdf` |
| 2 | 2022 | PROTEIN AI Advisor | Safety-first personalized meal planning | `papers/02_2022_Stefanidis_PROTEIN_AI_Advisor.pdf` |
| 3 | 2022 | Relative Validation of an AI-Enhanced, Image-Assisted Mobile App | Human-centered food-logging validation | `papers/03_2022_Moyen_AI_Image_Assisted_Dietary_App.pdf` |
| 4 | 2023 | AI-Based Digital Image Dietary Assessment Methods Compared to Humans and Ground Truth | Accuracy and evidence-quality review | `papers/04_2023_Shonkoff_AI_Digital_Image_Dietary_Assessment_Review.pdf` |
| 5 | 2024 | OVFoodSeg: Elevating Open-Vocabulary Food Image Segmentation via Image-Informed Textual Representation | Open-vocabulary food segmentation | `papers/05_2024_Wu_OVFoodSeg_CVPR.pdf` |
| 6 | 2023 | The Nutritional Content of Meal Images in Free-Living Conditions—goFOOD | Free-living image-to-nutrient pipeline | `papers/06_2023_Papathanail_goFOOD_Free_Living.pdf` |
| 7 | 2023 | An Automated Image-Based Dietary Assessment System for Mediterranean Foods | Cuisine-specific recognition and stereo portion estimation | `papers/07_2023_Konstantakopoulos_MedGRFood_Mediterranean.pdf` |
| 8 | 2023 | Personalized Flexible Meal Planning for Individuals With Diet-Related Health Concerns | Constraint-aware recommendation | `papers/08_2023_Amiri_Personalized_Flexible_Meal_Planning.pdf` |
| 9 | 2025 | Are Vision-Language Models Ready for Dietary Assessment? | VLM food-recognition benchmark | `papers/09_2025_Romero_Tapiador_VLM_Dietary_Assessment_CVPRW.pdf` |
| 10 | 2024 | Visual Nutrition Analysis | RGB segmentation and direct nutrient regression | `papers/10_2024_Zhao_Visual_Nutrition_Analysis.pdf` |
| 11 | 2024 | AI Nutrition Recommendation Using a Deep Generative Model and ChatGPT | Guideline-constrained weekly plan generation | `papers/11_2024_Papastratis_AI_Nutrition_Recommendation.pdf` |
| 12 | 2024 | ChatDiet | Conversational personalized recommendation | `papers/12_2024_Yang_ChatDiet_Personalized_Recommender.pdf` |
| 13 | 2024 | Delighting Palates with AI | Reinforcement-learning meal recommendations | `papers/13_2024_Amiri_Reinforcement_Learning_Meal_Plans.pdf` |
| 14 | 2025 | DietAI24 | Multimodal LLM plus retrieval-grounded nutrition estimation | `papers/14_2025_Yan_DietAI24_Multimodal_Nutrition.pdf` |
| 15 | 2025 | An Evaluation of ChatGPT for Nutrient Content Estimation | Negative baseline for ungrounded ChatGPT | `papers/15_2025_OHara_ChatGPT_Nutrient_Estimation.pdf` |
| 16 | 2025 | Decomposing Food Images for Better Nutrition Analysis | Two-step multimodal nutrition analysis | `papers/16_2025_Khlaisamniang_Decomposing_Food_Images_CVPRW.pdf` |
| 17 | 2025 | MOPI-HFRS: A Multi-Objective Personalized Health-Aware Food Recommendation System With LLM-Enhanced Interpretation | Health-aware multi-objective recommendation | `papers/17_2025_Zhang_MOPI_HFRS_NSF_Accepted_Manuscript.pdf` |
| 18 | 2026 | Development of an Intelligent Food Nutrition Recognition and Nutrient Intake Assessment System in Hospital Settings | Clinical RGB-D and residual-intake system | `papers/18_2026_Wang_Hospital_Food_Nutrition_System.pdf` |
| 19 | 2026 | Model Architecture Dominates Nutritional Estimation Accuracy in VLM Systems | Broad VLM benchmark and negative evidence | `papers/19_2026_Vedovelli_VLM_Nutritional_Estimation_Nature.pdf` |
| 20 | 2026 | Limited Validity of an AI-Powered App for Dietary Assessment in Females With Obesity | Physiological validation and safety warning | `papers/20_2026_Serra_SNAQ_Limited_Validity_Nature.pdf` |

---

## 4. Paper-by-paper critical analysis

### 4.1 Applying Image-Based Food-Recognition Systems on Dietary Assessment: A Systematic Review

**Source:** K. V. Dalakleidi, M. Papadelli, I. Apostolos, and K. Papadimitriou, *Advances in Nutrition*, 13(6), 2590–2619, 2022. DOI: [10.1093/advances/nmac078](https://doi.org/10.1093/advances/nmac078)

**Methodology:** The review screened 159 publications and included 78 studies. It organizes image-based dietary assessment into image capture, segmentation, feature extraction, classification, volume estimation, and calorie/nutrient calculation. It compares hand-crafted features with CNNs and summarizes 27 public food-image datasets.

**Key contributions and results:** Forty-five of 78 studies used deep learning. CNNs generally outperformed shallow methods on large public datasets; Food-101 accuracy ranged from 55.3% to 90.27% across reviewed approaches. The paper shows that classification is the most developed stage, while volume and nutrient estimation remain weak and under-evaluated.

**Advantages:** Best system-level map for the project. It identifies appropriate modules, datasets, metrics, and common failures such as occlusion, mixed foods, lighting, leftovers, and limited cuisine coverage.

**Limitations:** Studies use incompatible datasets, cameras, metrics, and ground truths, so the review cannot provide a fair pooled model ranking. Dataset search ended in 2021, and it predates recent vision-language models.

**Project use:** Use it to define the analyzer pipeline and literature-search taxonomy. Do not treat high food-classification accuracy as proof of calorie or clinical accuracy.

---

### 4.2 PROTEIN AI Advisor: A Knowledge-Based Recommendation Framework Using Expert-Validated Meals for Healthy Diets

**Source:** K. Stefanidis et al., *Nutrients*, 14(20), 4435, 2022. DOI: [10.3390/nu14204435](https://doi.org/10.3390/nu14204435)

**Methodology:** A two-layer expert system first uses a nutrition ontology and fuzzy rule reasoning to reject meals that conflict with safety or appropriateness constraints such as allergies, medical conditions, and dietary choices, while promoting nutrient-relevant foods. A second layer quantitatively optimizes energy, macro- and micronutrients, meal variety, and daily/weekly plans. The meal database contains 2,266 expert-validated meals, and testing covered 3,000 virtual profiles.

**Key contributions and results:** The approach separates hard safety rules from softer optimization. Mean meal-rejection F1 was 0.995. Reported plan accuracy was 92.65% for macronutrients and 85.86% for other components. Generated-versus-target energy had a reported difference of +7.32% with Pearson `r = 0.983`.

**Advantages:** Transparent, auditable, and suitable for allergies and medical constraints. It is easier to explain than an unconstrained neural recommender.

**Limitations:** The 92% result is agreement with expert rules on synthetic profiles, not improved adherence or clinical outcomes. The database has limited coverage for some diets, athletes, complex allergies, and cultures.

**Project use:** Adopt the ontology/rule layer as a safety gate after image analysis. Never let an LLM override a hard allergy or medical constraint.

---

### 4.3 Relative Validation of an Artificial Intelligence–Enhanced, Image-Assisted Mobile App for Dietary Assessment in Adults

**Source:** A. Moyen, A. I. Rappaport, C. Fleurent-Grégoire, A.-J. Tessier, A.-S. Brazeau, and S. Chevalier, *Journal of Medical Internet Research*, 24(11), e40449, 2022. DOI: [10.2196/40449](https://doi.org/10.2196/40449)

**Methodology:** Keenoa combines food photographs, AI recognition, user portion confirmation, pictograms, barcode lookup, food-composition databases, and dietitian review. A randomized crossover study compared four days of Keenoa with four days of ASA24 in 136 adults, including 31 with diabetes.

**Key contributions and results:** Mean energy difference was −32 kcal/day, but limits of agreement were −789 to 725 kcal/day. Nutrient correlations ranged from 0.40 to 0.74, while individual cross-classification was fair to moderate. Usability was 77/100 for Keenoa versus 53/100 for ASA24, and 74.8% preferred Keenoa.

**Advantages:** Strong randomized design, clinical population, usability evidence, and support for a recognize-then-confirm workflow.

**Limitations:** Users and dietitians corrected entries, so autonomous vision performance was not isolated. The comparator was not a gold standard, individual agreement was limited, and the study was short.

**Project use:** Use human confirmation, barcode fallback, culturally broad databases, and explicit correction controls. Design for adherence, not just model accuracy.

---

### 4.4 AI-Based Digital Image Dietary Assessment Methods Compared to Humans and Ground Truth: A Systematic Review

**Source:** E. Shonkoff et al., *Annals of Medicine*, 55(2), 2273497, 2023. DOI: [10.1080/07853890.2023.2273497](https://doi.org/10.1080/07853890.2023.2273497)

**Methodology:** The review screened 14,059 publications and included 52 fully automated image-assessment studies through May 2023. It normalized results to relative error where possible and compared food volume, energy, and nutrient outcomes with human estimates and study-defined ground truth.

**Key contributions and results:** CNNs appeared in 79% of papers. Ground truth was nutrient-table calculation in 51% and weighed food in 27%; no included study used bomb calorimetry. Average calorie errors ranged from 0.10% to 38.3%, and average volume errors from 0.09% to 33%. Simple single-food images generally produced lower errors than mixed meals.

**Advantages:** Directly evaluates end-to-end dietary measurement rather than classification alone and provides a strong validation agenda.

**Limitations:** Heterogeneity prevented meta-analysis. Results often reflect favorable datasets, best-performing model versions, or database-derived rather than measured ground truth.

**Project use:** Report absolute and relative errors, volume error separately from nutrient error, use diverse held-out meals, and compare with weighed food or other strong reference methods.

---

### 4.5 OVFoodSeg: Elevating Open-Vocabulary Food Image Segmentation via Image-Informed Textual Representation

**Source:** X. Wu, S. Yu, E.-P. Lim, and C.-W. Ngo, *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 4144–4153, 2024. DOI: [10.1109/CVPR52733.2024.00397](https://doi.org/10.1109/CVPR52733.2024.00397)

**Methodology:** OVFoodSeg addresses open-vocabulary ingredient segmentation. A FoodLearner image-to-text module aligns food images with food-specific textual representations, while an Image-Informed Text Encoder adds image context to the text embedding. Training has a FoodLearner pretraining stage followed by segmentation adaptation.

**Key contributions and results:** The abstract and main text report a 4.9% mIoU improvement on FoodSeg103 (and 3.5% on FoodSeg195) over SAN. However, Table 1’s mean novel-class mIoU values (SAN 19.0% versus OVFoodSeg 33.6%) imply a roughly 14.6-point absolute gain, so the paper’s percentage wording and table are not fully reconciled. The result nevertheless indicates better handling of novel or diverse ingredients than closed-vocabulary segmentation baselines.

**Advantages:** Open-set ingredient recognition, reduced dependence on fixed category lists, and direct relevance to mixed-meal segmentation.

**Limitations:** It is a segmentation method, not a calorie or nutrient estimator; benchmark performance does not establish portion accuracy; visually similar ingredients and dataset bias remain concerns.

**Project use:** Use as the segmentation component or as the open-set fallback when a closed classifier is uncertain. Add portion estimation, composition lookup, and user correction afterward.

---

### 4.6 The Nutritional Content of Meal Images in Free-Living Conditions—Automatic Assessment with goFOOD™

**Source:** I. Papathanail et al., *Nutrients*, 15(17), 3835, 2023. DOI: [10.3390/nu15173835](https://doi.org/10.3390/nu15173835)

**Methodology:** goFOOD combines Mask R-CNN food segmentation, hierarchical RegNetY recognition, monocular Zoe depth, reference-card geometry, portion estimation, barcode lookup, and nutrition databases. A free-living study involved 49 analyzable adults, 444 images, and 587 food items.

**Key contributions and results:** Fine-category top-1 recognition was 58.7%, with 78.8% top-5. Single-image MAPE was 27.41% for calories, 31.27% for carbohydrate, 39.17% for protein, and 43.24% for fat. The study also collected user-experience data, with 83.3% reporting easy use.

**Advantages:** Integrates recognition, portion, database lookup, and a free-living evaluation without requiring a stereo pair or depth sensor for analysis.

**Limitations:** Small, narrow sample; one study day; dietitian recall rather than weighed ground truth; and optimistic top-five recognition assumptions. Hidden oil, cooking method, and ingredient composition remain major errors.

**Project use:** Use as a modular engineering reference, but add prompts for cooking oil, portion confirmation, uncertainty, and a longer prospective study.

---

### 4.7 An Automated Image-Based Dietary Assessment System for Mediterranean Foods

**Source:** F. S. Konstantakopoulos, E. I. Georga, and D. I. Fotiadis, *IEEE Open Journal of Engineering in Medicine and Biology*, 4, 45–54, 2023. DOI: [10.1109/OJEMB.2023.3266135](https://doi.org/10.1109/OJEMB.2023.3266135)

**Methodology:** MedGRFood includes 51,840 classification images across 160 classes and 10,000 stereo volume images. EfficientNetB2 performs classification; BEBLID matching, RANSAC, stereo triangulation, point clouds, and convex-hull reconstruction estimate volume. A reference card sets scale, USDA volume-to-weight data convert volume to weight, and the Greek Food Composition Database converts weight to macronutrients.

**Key contributions and results:** Classification reached 83.8% top-1 and 97.6% top-5 accuracy. Volume MAPE across 148 dishes was 10.5%. The work provides cuisine-specific coverage rarely found in generic datasets.

**Advantages:** Explicit 3D portion estimation, food-domain pretraining, and a complete recognition-to-nutrient architecture for Mediterranean foods.

**Limitations:** Two calibrated images, a reference card, shallow dishes, controlled distance/angle, and separate component evaluations reduce generality. End-to-end calorie and macronutrient error was not reported.

**Project use:** Adapt cuisine-specific data collection and 3D reconstruction, but simplify capture and validate complete nutrient output on free-living meals.

---

### 4.8 Personalized Flexible Meal Planning for Individuals With Diet-Related Health Concerns

**Source:** M. Amiri, J. Li, and W. Hasan, *JMIR Formative Research*, 7, e46434, 2023. DOI: [10.2196/46434](https://doi.org/10.2196/46434)

**Methodology:** The system uses a food/nutrition ontology, a biocultural user-profile ontology, semantic rules, fuzzy nutrient scoring, heuristic meal search, AHP weighting, and TOPSIS ranking. The recipe corpus contains 176,206 recipes and 563 food tags. A feasibility study involved 39 adults.

**Key contributions and results:** The planner integrates health conditions, allergies, religion, culture, cost, cooking time, cuisine, and sensory preferences. Thirty-eight of 39 users rated it helpful, and 37 of 39 reported time savings. App-generated plans had much better internal nutrition scores than participant-created plans.

**Advantages:** Clear separation of mandatory safety constraints and soft preferences; interpretable and broad personalization.

**Limitations:** Only one participant had diabetes and three had hypertension. The study lasted one week, used internal PV/TOPSIS scores, and did not report long-term adherence or health outcomes. Dynamic updates were not implemented.

**Project use:** Use as the recommendation-layer blueprint. Add adaptive updates, verified nutrient data, stronger clinical guardrails, and explainable reasons for every recommendation.

---

### 4.9 Are Vision-Language Models Ready for Dietary Assessment? Exploring the Next Frontier in AI-Powered Food Image Recognition

**Source:** S. Romero-Tapiador et al., *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, 430–439, 2025. DOI: [10.1109/CVPRW67362.2025.00047](https://doi.org/10.1109/CVPRW67362.2025.00047)

**Methodology:** The study evaluates six VLMs—ChatGPT, Gemini, Claude, Moondream, DeepSeek, and LLaVA—using FoodNExTDB. The dataset contains 9,263 expert-labeled images, 10 broad categories, 62 subcategories, nine cooking styles, and approximately 50,000 nutrition labels. The authors introduce Expert-Weighted Recall (EWR) to account for annotator variability.

**Key contributions and results:** Closed-source models exceeded 90% EWR for some single-product images, but performance dropped sharply with granularity: Gemini averaged 70.16% EWR across the three levels and fell to 50.00% when cooking style was included. This exposes substantial difficulty with fine-grained distinctions, visually similar foods, and cooking-style differences.

**Advantages:** Large expert-labeled benchmark, explicit evaluation of an important VLM failure mode, and a reproducible recognition-oriented test set.

**Limitations:** It is a CVPR workshop paper focused on recognition rather than portions or nutrient regression. Model versions are proprietary and can change; EWR is not clinical intake accuracy.

**Project use:** Use FoodNExTDB/EWR ideas for evaluation and error analysis. Do not treat high single-food recognition scores as evidence of nutrient accuracy.

---

### 4.10 Visual Nutrition Analysis: Leveraging Segmentation and Regression for Food Nutrient Estimation

**Source:** Y. Zhao, P. Zhu, Y. Jiang, and K. Xia, *Frontiers in Nutrition*, 11, 1469878, 2024. DOI: [10.3389/fnut.2024.1469878](https://doi.org/10.3389/fnut.2024.1469878)

**Methodology:** The authors manually segmented 3,224 Nutrition5k images. U-Net isolates food; ResNet-101 and Squeeze-and-Excitation features feed regression heads for mass, calories, fat, carbohydrate, and protein. The method uses measured ingredient weights and nutrient values calculated from those weights using USDA composition data rather than a food-class-to-fixed-nutrient lookup.

**Key contributions and results:** U-Net achieved 97.69% Dice. Mean PMAE across the five mass/nutrient output targets was 17.06%; calorie PMAE was 15.68%. The model used 51.96 million parameters and 10.28 GFLOPs, less than the compared RGB-D model.

**Advantages:** Direct multi-output nutrition regression, public segmentation labels, and no depth camera.

**Limitations:** Controlled single-source images, binary foreground segmentation, limited cuisine coverage, and no free-living or clinical validation. PMAE can hide small-target errors, and the paper contains an internal inconsistency in one reported mean PMAE value and an ambiguous ablation table.

**Project use:** Strong baseline for the analyzer, but add ingredient decomposition, external datasets, uncertainty, and mobile correction.

---

### 4.11 AI Nutrition Recommendation Using a Deep Generative Model and ChatGPT

**Source:** I. Papastratis, D. Konstantinidis, P. Daras, and K. Dimitropoulos, *Scientific Reports*, 14, 14620, 2024. DOI: [10.1038/s41598-024-65438-x](https://doi.org/10.1038/s41598-024-65438-x)

**Methodology:** A variational autoencoder encodes anthropometric and medical profiles; a GRU decoder generates six meals per day. Training losses encode EFSA/WHO energy and macronutrient constraints. A portion optimizer rescales meals to target energy. ChatGPT expands a validated meal pool with Greek, Italian, and Portuguese equivalents. Testing covered 3,000 virtual profiles and 1,000 external profiles.

**Key contributions and results:** The system reported average macronutrient accuracy of 86.99% on PROTEIN NAP and 84.19% on external profiles. It generated weekly plans in roughly 500 ms and reported cuisine-level average weekly meal-plan variability of 4.88–4.99 on a 1–7 scale.

**Advantages:** Guideline-constrained generation, fast inference, cuisine expansion, and explicit user-profile representation.

**Limitations:** Predominantly synthetic evaluation, no prospective adherence trial, and no independent clinical validation of ChatGPT-generated nutrient equivalents. The paper includes database-relative equivalence testing, but the reported zero energy difference largely follows the portion optimizer rather than real intake. Allergy safety and clinical outcomes were not validated.

**Project use:** Use the generative planner as an optional module after deterministic safety filtering and verified nutrient data are applied.

---

### 4.12 ChatDiet: Empowering Personalized Nutrition-Oriented Food Recommender Chatbots Through an LLM-Augmented Framework

**Source:** Z. Yang et al., *Smart Health*, 32, 100465, 2024. DOI: [10.1016/j.smhl.2024.100465](https://doi.org/10.1016/j.smhl.2024.100465)

**Methodology:** ChatDiet separates a personal causal model, a population nutrition model, a retrieval orchestrator, and an LLM. The personal model uses longitudinal wearable and nutrition data; the population model uses food composition knowledge; BM25 retrieves information for zero-shot LLM prompting.

**Key contributions and results:** The framework supports explanations and conversational follow-up. It reported a 92% recommendation-effectiveness ratio, assessed on 400 manually evaluated query-response pairs. The real personal-model evidence came from one N-of-1 case; most quantitative evaluation used synthetic data.

**Advantages:** Clear architecture for grounded personalization and explanations, and separation of personal effects from general nutrition knowledge.

**Limitations:** Causal inference from observational data is assumption-sensitive, the LLM can contradict retrieved facts, and the effectiveness metric is not clinical benefit. There was no held-out human cohort or prospective study.

**Project use:** Suitable for the explanation/chat layer, provided every response is grounded in verified personal and population data and exposes uncertainty.

---

### 4.13 Delighting Palates with AI: Reinforcement Learning’s Triumph in Crafting Personalized Meal Plans with High User Acceptance

**Source:** M. Amiri, F. S. Rad, and J. Li, *Nutrients*, 16(3), 346, 2024. DOI: [10.3390/nu16030346](https://doi.org/10.3390/nu16030346)

**Methodology:** Collaborative filtering produces user and meal embeddings; a reinforcement-learning agent selects meals using rewards from nutrition, preference, and user ratings. AHP/TOPSIS represent taste, cost, convenience, and other preferences. Experiments used more than 180,000 Food.com recipes and 700,000 interactions.

**Key contributions and results:** The reported CFRL method achieved a nutrition score of 0.90, preference score of 0.84, and acceptance of 4.5/5. However, collaborative filtering alone also received 4.5/5, while a nutrition-only baseline had higher nutrition and preference scores.

**Advantages:** Models taste and acceptance, supports feedback-based adaptation, and addresses adherence rather than nutrient compliance alone.

**Limitations:** Retrospective ratings and a simulated use case do not prove real acceptance. Cold start, allergy safety, and clinical portion constraints are unresolved. The paper also contains inconsistent descriptions of the RL algorithm and hyperparameters.

**Project use:** Use feedback and preference learning only after safe candidates are generated and verified. Do not claim that the reported acceptance equals adherence.

---

### 4.14 DietAI24 as a Framework for Comprehensive Nutrition Estimation Using Multimodal Large Language Models

**Source:** R. Yan et al., *Communications Medicine*, 5, 458, 2025. DOI: [10.1038/s43856-025-01159-0](https://doi.org/10.1038/s43856-025-01159-0)

**Methodology:** GPT Vision creates an image description and retrieval queries. LangChain retrieves matching FNDDS food descriptions, and GPT-4 Vision/Turbo selects candidate food codes and portion descriptors. FNDDS maps portions to grams and supplies 65 nutrient/component vectors, which the system sums arithmetically. Evaluation used 3,000 ASA24 images and 1,000 Nutrition5k images.

**Key contributions and results:** On ASA24, the system reported energy MAE of 47.7 kcal, fat MAE of 1.8 g, carbohydrate MAE of 6.95 g, and protein MAE of 1.38 g. On Nutrition5k, it reported energy MAE of 68.2 kcal. It is zero-shot and grounded in a standardized nutrient database.

**Advantages:** The closest selected paper to a practical image-to-food-code-to-nutrient pipeline. RAG reduces reliance on LLM memory and supports regional database replacement.

**Limitations:** Only selected outputs had quantitative ground truth; U.S. datasets dominate; and results depend on closed OpenAI models. Database labels do not prove real-world portion or clinical accuracy.

**Project use:** This is the strongest reference for the measurement module. Retain the retrieval architecture, but add local food databases, uncertainty, and non-OpenAI fallback options.

---

### 4.15 An Evaluation of ChatGPT for Nutrient Content Estimation from Meal Photographs

**Source:** C. O’Hara, G. Kent, A. C. Flynn, E. R. Gibney, and C. M. Timon, *Nutrients*, 17(4), 607, 2025. DOI: [10.3390/nu17040607](https://doi.org/10.3390/nu17040607)

**Methodology:** ChatGPT-4 was used without fine-tuning or an external food database to identify foods, estimate weights, and report 16 nutrients from 114 photographs of 38 weighed Irish survey meals. Seven dietitians estimated selected nutrients for comparison.

**Key contributions and results:** Food recognition was strong at 88.6% F1, but ChatGPT underestimated meal weight in 76.3% of photographs. Mean absolute weight difference was 27.8%. The abstract reports that 13 of 16 nutrients differed by more than 10%, while the detailed results and table support 12; mean absolute nutrient difference was 26.9%.

**Advantages:** Objective weighed meals, multiple portion sizes, broad nutrient testing, and a direct dietitian comparison.

**Limitations:** Small, culturally narrow, single-model, single-prompt evaluation without database grounding. A non-significant mean difference was sometimes interpreted as agreement, but that does not prove agreement.

**Project use:** Use as evidence that raw LLM output is not a trustworthy nutrient source. Pair recognition with portion calibration, food-database retrieval, and conservative uncertainty handling.

---

### 4.16 Decomposing Food Images for Better Nutrition Analysis: A Nutritionist-Inspired Two-Step Multimodal LLM Approach

**Source:** P. Khlaisamniang et al., *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, 482–491, 2025. DOI: [10.1109/CVPRW67362.2025.00053](https://doi.org/10.1109/CVPRW67362.2025.00053)

**Methodology:** The framework uses an off-the-shelf multimodal LLM in two steps. The first step decomposes a dish into ingredients, portion sizes, and cooking details; the second step converts that structured interpretation into calories and macronutrients. Evaluation uses a Nutrition5k subset (Nutrition320) and 121 real-world Gindee samples, with additional bounding-box and segmentation-mask prompts.

**Key contributions and results:** The two-step decomposition outperformed direct one-step querying in most tested settings. With GPT-4o, it reduced calorie error to 78.49 kcal on Nutrition320 (12.11% better than baseline). On Gindee121, the paper’s prose reports a 10% improvement to 85.5 kcal, while its tabulated values imply approximately a 12.4% reduction; Qwen2.5-VL with bounding boxes was slightly better on Gindee121. Separating “what is on the plate?” from “how does it translate nutritionally?” improves interpretability and, in most tests, accuracy.

**Advantages:** Low training burden, interpretable intermediate outputs, and a practical prompt pipeline that mirrors nutritionist reasoning.

**Limitations:** It is a CVPR workshop paper, relies on proprietary MLLMs, does not establish clinical validity, and cannot reliably infer hidden ingredients or exact cooking oil. Reported gains are benchmark/application results rather than health outcomes.

**Project use:** Use the two-stage structure as a user-facing explanation flow, but verify the intermediate ingredient and portion fields against a food database and explicit uncertainty prompts.

---

### 4.17 MOPI-HFRS: A Multi-Objective Personalized Health-Aware Food Recommendation System With LLM-Enhanced Interpretation

**Source:** Z. Zhang et al., *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1*, 2860–2871, 2025. DOI: [10.1145/3690624.3709382](https://doi.org/10.1145/3690624.3709382)

**Methodology:** NHANES 2003–2020 data are converted into two health-aware recommendation benchmarks. Signed graph learning incorporates health information; Pareto optimization balances preference, personalized healthiness, and nutritional diversity; an LLM produces explanations from learned graph context.

**Key contributions and results:** The full-nutrient benchmark contains 13,282 users, 7,516 foods, and 488,223 interactions. The system outperformed the tested graph and recommender baselines across reported ranking, health, and diversity measures.

**Advantages:** Integrates medical information into training, makes multi-objective trade-offs explicit, and connects ranking with explanations.

**Limitations:** Healthiness is encoded using fixed nutrient thresholds, and explanation quality is measured with BERT/BLEU similarity rather than clinical factuality. There is one data source, no clearly user-disjoint split, and no prospective or clinical validation.

**Project use:** Use graph ranking as a recommendation backend, but keep deterministic allergy/medical rules outside the learned model and require clinician oversight for high-risk profiles.

---

### 4.18 Development of an Intelligent Food Nutrition Recognition and Nutrient Intake Assessment System in Hospital Settings

**Source:** B. Wang et al., *Frontiers in Nutrition*, 13, 1637034, 2026. DOI: [10.3389/fnut.2026.1637034](https://doi.org/10.3389/fnut.2026.1637034)

**Methodology:** The hospital system uses standardized meal boxes, depth imaging, marker-guided SAM segmentation, SE_ResNet50_vd classification, 3D volume reconstruction, electronic weighing, and Chinese nutrient/composition databases. Pre- and post-meal measurements estimate consumed quantity.

**Key contributions and results:** Food recognition reached 99.2% on 1,000 tests. Across 100 food items, 39% exceeded 90% estimated/measured volume ratio and 61% were between 80% and 90%. The system covered more than 20 therapeutic-diet types.

**Advantages:** A controlled hospital workflow, physically grounded portions, multiple-dish segmentation, and auditable nutrient composition data.

**Limitations:** Closed standardized menus, depth hardware, and pre-meal validation limit general use. Post-meal leftovers, segmentation metrics, end-to-end nutrient error, and patient outcomes were not validated. Volume summaries are internally inconsistent and use a ratio that can reward systematic underestimation.

**Project use:** Adopt marker-guided capture, reference weighing, and pre/post residual logic for controlled environments, but not the claim of unrestricted recognition.

---

### 4.19 Model Architecture Dominates Nutritional Estimation Accuracy in Vision-Language Systems

**Source:** L. Vedovelli et al., *Scientific Reports*, vol. 16, art. 29227, 2026. DOI: [10.1038/s41598-026-58755-w](https://doi.org/10.1038/s41598-026-58755-w)

**Methodology:** Forty VLMs from eight providers underwent initial screening on Nutrition5k. The paper reports 9,050 initial predictions, although its stated 40-model × 50-dish × 5-condition design implies 10,000. Nine selected models were then evaluated across image counts, temperatures, and seven prompt types in additional experiments. Energy, carbohydrate, protein, and fat were assessed using RMSLE, sMAPE, MAE, and nonparametric tests.

**Key contributions and results:** Model architecture accounted for a reported 99.6% of measured variance. Configurations using one, two, three, or five images did not significantly change performance, and complex prompts did not significantly outperform a baseline. High-quality consumer images outperformed controlled images, and ingredient descriptions modestly improved aggregate performance. The professional nutritionist substantially outperformed the best AI configuration.

**Advantages:** Broad model comparison, repeated conditions, corrected statistics, and useful negative evidence against unnecessary multi-angle capture or prompt complexity.

**Limitations:** The core experiments sampled 50 dishes, but the paper’s reported prediction counts are internally inconsistent: 40 × 50 × 5 implies 10,000 rather than 9,050, and the multi-image analysis reports 1,765 rather than the implied 1,800. High-quality comparisons use varying sample counts (16–23, with the abstract implying 22) and a seven-image professional benchmark, so cross-condition comparisons should be treated as exploratory. Architecture and provider are partly confounded. The best general VLM still had approximately 50% aggregate symmetric error.

**Project use:** Benchmark candidate models early, use a simple single-image path, collect ingredient metadata when possible, and do not claim clinical accuracy from general VLMs.

---

### 4.20 Limited Validity of an AI-Powered App for Dietary Assessment in Females With Obesity

**Source:** M. Serra et al., *npj Digital Medicine*, 9, 357, 2026. DOI: [10.1038/s41746-026-02536-2](https://doi.org/10.1038/s41746-026-02536-2)

**Methodology:** SNAQ uses image/depth sensing and a food database. Twenty women with obesity completed a seven-day free-living study. Estimated intake was compared with doubly labelled water and one 24-hour recall using bias, limits of agreement, correlation, plausibility, and reliability analyses.

**Key contributions and results:** SNAQ underestimated energy by 817 kcal/day, or 25%, relative to doubly labelled water. Limits of agreement ranged from −3,707 to +2,073 kcal/day. Association was negligible (`R² = 0.01`) and within-person reliability was `ICC = 0.00`.

**Advantages:** Strong physiological reference, clinically important population, multidimensional validity analysis, and transparent disclosure of developer interest.

**Limitations:** Small female-only sample, proprietary algorithm, incomplete adherence monitoring, and use of energy expenditure as an intake proxy under energy-balance assumptions.

**Project use:** Treat this as the safety gate. Before recommendations are affected, require subgroup testing, bias and limits-of-agreement thresholds, reliability, adherence telemetry, and escalation for uncertain records.

---

## 5. Cross-paper synthesis

### 5.1 Recognition accuracy is not nutrition accuracy

Food classification can exceed 90% while calorie or nutrient errors remain large. Papers 5, 9, and 16 demonstrate useful visual or multimodal analysis, but none provides sufficient evidence for nutrient precision. Papers 6, 7, 10, 14, and 15 show that portion estimation, ingredient composition, hidden oil, and food-database mapping dominate the final error.

### 5.2 Portion and ingredient uncertainty remain the main technical bottleneck

Single-image depth, stereo reconstruction, reference cards, electronic weighing, and VLM reasoning all reduce or expose portion error, but none removes it. Hidden fats, cooking losses, mixed dishes, occluded ingredients, and plate size cannot be reliably inferred from appearance alone.

### 5.3 General-purpose LLMs should be grounded, corrected, and constrained

Paper 9 shows that VLM food recognition depends strongly on the model and on fine-grained task difficulty. Paper 15 shows systematic underestimation from raw ChatGPT. Paper 14 shows why retrieval over a food-composition database is more credible than asking an LLM to remember nutrient values. The correct design is therefore `vision model + verified database + explicit uncertainty`, not `LLLM-only nutrient estimation`.

### 5.4 Human-in-the-loop design is currently necessary

Keenoa, goFOOD, the hospital system, and the two clinical/validation studies all indicate that correction, dietitian review, standardized procedures, or clinical oversight materially affect results. A fully autonomous analyzer is not supported by the current evidence.

### 5.5 Personalization should be downstream of measurement

The recommendation papers use structured nutrients, user profiles, health conditions, preferences, and food knowledge. If the meal-analysis layer is wrong, personalization becomes confidently wrong. The strongest architecture separates:

`image/portion estimate → verified nutrients → hard safety rules → personalized ranking/explanation`.

### 5.6 Offline scores are not health outcomes

High classification accuracy, recommendation effectiveness, internal PV scores, acceptance scores, or nutrient correlation do not establish adherence, safety, or improved health. Only a small subset of papers used objective intake references, and very few measured clinical outcomes.

---

## 6. Comparative assessment of method families

| Method family | Representative papers | Advantages | Main disadvantages | Best role in this project |
|---|---|---|---|---|
| CNN/Transformer food recognition | 5, 6, 7 | Strong image classification and segmentation; efficient local deployment possible | Needs labeled domain data; weak at unseen foods, mixtures, and portion estimation | Visual feature extractor or recognition stage |
| RGB-D/stereo portion estimation | 6, 7, 18 | Physically grounded size and volume; better than pure area heuristics | Hardware/calibration burden; plate/cuisine/lighting sensitivity | Optional calibration mode or controlled-environment mode |
| Monocular/multi-modal portion reasoning | 6, 9, 19 | Ordinary smartphone capture; deployable | Absolute scale, occlusion, and hidden ingredients remain ambiguous | Lightweight portion estimate with uncertainty |
| General VLM/LLM estimation | 9, 15, 19 | Fast zero-shot reasoning, natural-language correction, broad output | Hallucination, closed versions, prompt sensitivity, inaccurate absolute nutrients | Interface, candidate recognition, and explanation—not final nutrient authority |
| MLLM plus retrieval/database | 14 | Grounds values in standardized composition data; supports regional replacement | Database coverage and portion mapping still determine accuracy | Preferred nutrient-calculation backbone |
| Two-step MLLM decomposition | 16 | Exposes ingredients, portions, and cooking assumptions before nutrient calculation | Still depends on MLLM judgment; hidden ingredients and oil remain unresolved | Interpretable estimation and correction flow |
| Knowledge-based recommendation | 2, 8 | Transparent constraints, strong safety, explainable | Requires expert rules and maintained databases; less flexible | Hard safety and eligibility layer |
| Generative/LLM recommendation | 11, 12 | Fast natural-language plans and interaction | Hallucination, synthetic evaluation, weak clinical evidence | Optional plan/chat layer after filtering |
| Collaborative filtering/RL | 13 | Models taste and feedback; the study did not show an incremental acceptance gain over collaborative filtering alone | Cold start, feedback bias, limited safety, no clinical validation | Preference-ranking layer |
| Health-aware graph recommendation | 17 | Uses medical data and explicit multi-objective trade-offs | Tag proxies, offline evaluation, limited clinical factuality | Secondary ranking/explanation backend |
| Clinical/physiological validation | 18, 20 | Stronger real-world or objective evidence | Expensive, small samples, difficult reproducibility | Required evaluation protocol and safety threshold |

---

## 7. Recommended architecture for the B.Tech project

### 7.1 Input and quality gate

- One normal food image, with an optional reference card or depth capture.
- Automatic blur, lighting, framing, and occlusion checks.
- Immediate user prompt when image quality is inadequate.
- Optional before/after image to improve consumed-portion estimation.

### 7.2 Perception

- Food/ingredient segmentation using U-Net, SAM, or an efficient detector.
- Hierarchical food classification covering cuisine-specific categories.
- Ingredient and preparation cues: fried, oil visible, sauce, cheese, dressing, preparation method.
- Open-set fallback that asks the user to select a nearby item rather than forcing a wrong label.

### 7.3 Portion and nutrient layer

- Direct multi-output nutrition regression as in paper 10.
- Optional monocular-depth or stereo module as in papers 6 and 7.
- Verified local food-composition database or retrieval layer as in paper 14.
- Prediction intervals or confidence categories for calories, mass, and macronutrients.
- Mandatory user confirmation for low-confidence or high-risk nutrient estimates.

### 7.4 Safety and recommendation

- Ontology and hard-rule engine for allergies, religious restrictions, medical conditions, and nutrient ranges, based on papers 2 and 8.
- Optional generative planner for variety and explanations, based on papers 11 and 12.
- Preference ranking using collaborative filtering, reinforcement learning, or graph recommendation only after safe candidates are generated.
- Every recommendation should display the nutrient rationale and the estimated meal confidence.

### 7.5 Validation plan

The project should report:

- Food-recognition precision, recall, and F1.
- Segmentation Dice/IoU.
- Portion MAE/MAPE.
- Calorie and nutrient MAE/MAPE separately.
- Bland–Altman bias and limits of agreement.
- Confidence calibration.
- Performance by cuisine, meal complexity, lighting, and portion size.
- User correction rate and completion/dropout rate.
- Recommendation constraint-violation rate and user acceptance.

For a B.Tech project, a held-out test set plus a small user study is sufficient. A larger clinical claim would require weighed records or another strong reference and should not be claimed without that evidence.

---

## 8. Research gaps and defensible project contributions

The selected literature still leaves important gaps:

1. No selected paper validates the complete path from free-living meal image to safe personalized recommendation and improved user outcome.
2. General VLMs remain inaccurate for absolute nutrient values.
3. Hidden ingredients, oils, preparation methods, and mixed dishes remain poorly measured.
4. Public datasets are geographically and culturally narrow.
5. Recommendation studies mainly use synthetic profiles, offline ranking, or internal scores.
6. Human correction is common but rarely quantified as a formal model component.
7. Uncertainty calibration and safe escalation are rarely implemented.
8. User adherence and explanation quality are weakly validated.

Potential project contributions are therefore:

- A confidence-aware image-to-nutrient pipeline rather than a recognition-only demo.
- Cuisine-specific fine-grained ingredient recognition with user correction.
- Retrieval-grounded nutrient calculation rather than LLM-generated nutrient values.
- A transparent constraint engine separating safety from preference.
- A side-by-side evaluation showing the accuracy gain from reference cards, depth, ingredient prompts, and database retrieval.
- A clear human-in-the-loop safety gate for uncertain or high-risk estimates.

---

## 9. Conclusion

The strongest project design is not a single monolithic AI model. The literature supports a layered system: a vision model for segmentation and recognition; a calibrated portion estimator; a verified food-composition database; a rule-based safety engine; a personalized ranking or planning layer; and a human correction interface.

The evidence also gives clear warnings. High classification accuracy does not guarantee accurate calories. Raw ChatGPT nutrient estimates are unreliable. Database-grounded multimodal systems are more credible but still depend on portion calibration and database coverage. Personalization should occur only after measurement and safety checks. Finally, any health-related recommendation must communicate uncertainty and avoid unsupported clinical claims.

---

## 10. Verified references

1. K. V. Dalakleidi, M. Papadelli, I. Apostolos, and K. Papadimitriou, “Applying Image-Based Food-Recognition Systems on Dietary Assessment: A Systematic Review,” *Advances in Nutrition*, vol. 13, no. 6, pp. 2590–2619, 2022. doi: [10.1093/advances/nmac078](https://doi.org/10.1093/advances/nmac078).

2. K. Stefanidis et al., “PROTEIN AI Advisor: A Knowledge-Based Recommendation Framework Using Expert-Validated Meals for Healthy Diets,” *Nutrients*, vol. 14, no. 20, art. 4435, 2022. doi: [10.3390/nu14204435](https://doi.org/10.3390/nu14204435).

3. A. Moyen, A. I. Rappaport, C. Fleurent-Grégoire, A.-J. Tessier, A.-S. Brazeau, and S. Chevalier, “Relative Validation of an Artificial Intelligence–Enhanced, Image-Assisted Mobile App for Dietary Assessment in Adults: Randomized Crossover Study,” *Journal of Medical Internet Research*, vol. 24, no. 11, art. e40449, 2022. doi: [10.2196/40449](https://doi.org/10.2196/40449).

4. E. Shonkoff, K. C. Cara, X. Pei, M. C. Chung, S. Kamath, K. Panetta, and E. Hennessy, “AI-based digital image dietary assessment methods compared to humans and ground truth: A systematic review,” *Annals of Medicine*, vol. 55, no. 2, art. 2273497, 2023. doi: [10.1080/07853890.2023.2273497](https://doi.org/10.1080/07853890.2023.2273497).

5. X. Wu, S. Yu, E.-P. Lim, and C.-W. Ngo, “OVFoodSeg: Elevating Open-Vocabulary Food Image Segmentation via Image-Informed Textual Representation,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 4144–4153, 2024. doi: [10.1109/CVPR52733.2024.00397](https://doi.org/10.1109/CVPR52733.2024.00397).

6. I. Papathanail, L. A. Rahman, L. Brigato, N. S. Bez, M. F. Vasiloglou, K. van der Horst, and S. M. Mougiakakou, “The Nutritional Content of Meal Images in Free-Living Conditions—Automatic Assessment with goFOOD™,” *Nutrients*, vol. 15, no. 17, art. 3835, 2023. doi: [10.3390/nu15173835](https://doi.org/10.3390/nu15173835).

7. F. S. Konstantakopoulos, E. I. Georga, and D. I. Fotiadis, “An Automated Image-Based Dietary Assessment System for Mediterranean Foods,” *IEEE Open Journal of Engineering in Medicine and Biology*, vol. 4, pp. 45–54, 2023. doi: [10.1109/OJEMB.2023.3266135](https://doi.org/10.1109/OJEMB.2023.3266135).

8. M. Amiri, J. Li, and W. Hasan, “Personalized Flexible Meal Planning for Individuals With Diet-Related Health Concerns: System Design and Feasibility Validation Study,” *JMIR Formative Research*, vol. 7, art. e46434, 2023. doi: [10.2196/46434](https://doi.org/10.2196/46434).

9. S. Romero-Tapiador et al., “Are Vision-Language Models Ready for Dietary Assessment? Exploring the Next Frontier in AI-Powered Food Image Recognition,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, pp. 430–439, 2025. doi: [10.1109/CVPRW67362.2025.00047](https://doi.org/10.1109/CVPRW67362.2025.00047).

10. Y. Zhao, P. Zhu, Y. Jiang, and K. Xia, “Visual nutrition analysis: Leveraging segmentation and regression for food nutrient estimation,” *Frontiers in Nutrition*, vol. 11, art. 1469878, 2024. doi: [10.3389/fnut.2024.1469878](https://doi.org/10.3389/fnut.2024.1469878).

11. I. Papastratis, D. Konstantinidis, P. Daras, and K. Dimitropoulos, “AI nutrition recommendation using a deep generative model and ChatGPT,” *Scientific Reports*, vol. 14, art. 14620, 2024. doi: [10.1038/s41598-024-65438-x](https://doi.org/10.1038/s41598-024-65438-x).

12. Z. Yang, E. Khatibi, N. Nagesh, M. Abbasian, I. Azimi, R. Jain, and A. M. Rahmani, “ChatDiet: Empowering personalized nutrition-oriented food recommender chatbots through an LLM-augmented framework,” *Smart Health*, vol. 32, art. 100465, 2024. doi: [10.1016/j.smhl.2024.100465](https://doi.org/10.1016/j.smhl.2024.100465).

13. M. Amiri, F. S. Rad, and J. Li, “Delighting Palates with AI: Reinforcement Learning’s Triumph in Crafting Personalized Meal Plans with High User Acceptance,” *Nutrients*, vol. 16, no. 3, art. 346, 2024. doi: [10.3390/nu16030346](https://doi.org/10.3390/nu16030346).

14. R. Yan et al., “DietAI24 as a framework for comprehensive nutrition estimation using multimodal large language models,” *Communications Medicine*, vol. 5, art. 458, 2025. doi: [10.1038/s43856-025-01159-0](https://doi.org/10.1038/s43856-025-01159-0).

15. C. O’Hara, G. Kent, A. C. Flynn, E. R. Gibney, and C. M. Timon, “An Evaluation of ChatGPT for Nutrient Content Estimation from Meal Photographs,” *Nutrients*, vol. 17, no. 4, art. 607, 2025. doi: [10.3390/nu17040607](https://doi.org/10.3390/nu17040607).

16. P. Khlaisamniang et al., “Decomposing Food Images for Better Nutrition Analysis: A Nutritionist-Inspired Two-Step Multimodal LLM Approach,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, pp. 482–491, 2025. doi: [10.1109/CVPRW67362.2025.00053](https://doi.org/10.1109/CVPRW67362.2025.00053).

17. Z. Zhang et al., “MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation,” in *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1*, pp. 2860–2871, 2025. doi: [10.1145/3690624.3709382](https://doi.org/10.1145/3690624.3709382).

18. B. Wang et al., “Development of an intelligent food nutrition recognition and nutrient intake assessment system in hospital settings,” *Frontiers in Nutrition*, vol. 13, art. 1637034, 2026. doi: [10.3389/fnut.2026.1637034](https://doi.org/10.3389/fnut.2026.1637034).

19. L. Vedovelli et al., “Model architecture dominates nutritional estimation accuracy in vision-language systems,” *Scientific Reports*, vol. 16, art. 29227, 2026. doi: [10.1038/s41598-026-58755-w](https://doi.org/10.1038/s41598-026-58755-w).

20. M. Serra et al., “Limited validity of an AI-powered app for dietary assessment in females with obesity,” *npj Digital Medicine*, vol. 9, art. 357, 2026. doi: [10.1038/s41746-026-02536-2](https://doi.org/10.1038/s41746-026-02536-2).
