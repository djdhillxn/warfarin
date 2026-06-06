# Contextual Warfarin Dose Learning

This repository studies warfarin dose selection as an online contextual-bandit problem. Following the [project specification](data/default_project.pdf) and the [IWPC supplementary appendix](data/appx.pdf), each patient is represented by demographic, clinical, medication, and genetic covariates, and the learner chooses one of three weekly-dose classes: low (below 21 mg), medium (21–49 mg), or high (above 49 mg). The appropriate class is then revealed as feedback, allowing the policy to update before the next patient. Experiments report dosing accuracy, fraction incorrect, reward, and cumulative regret over repeated random patient orderings rather than a single favorable sequence.

## Data and preprocessing

The source [IWPC dataset](data/warfarin.csv) contains 5,700 patients, of whom 5,528 have a known stable therapeutic dose. Its central difficulty is not merely encoding mixed clinical data, but deciding what missingness means. The active pipeline in [preprocess_v2.ipynb](src/preprocessing/preprocess_v2.ipynb) removes patients without a target dose, excludes identifiers and post-treatment outcomes, preserves unknown categorical states where they carry information, derives medication and comorbidity indicators from text, parses indications and target-INR fields, constructs genotype allele counts, and applies the appendix S4 rules for imputing VKORC1 rs9923231. Formula-derived dose predictions are deliberately kept out of the bandit feature matrix.

A detailed account of every cleaning, imputation, encoding, leakage-control, and baseline-preparation decision is available in the [data-cleaning and feature-preparation report](src/preprocessing/DATA_CLEANING_AND_FEATURE_PREPARATION.md).

Height and weight receive particular attention because both are frequently incomplete and directly enter the established dosing equations. The KNN variant standardizes height and weight and imputes them from their two-dimensional neighborhood; it is retained as a direct refinement of the original preprocessing. The regression variant uses iterative Bayesian-ridge imputation informed by age, sex, race, smoking, diabetes, heart failure, and related patient context, so it can remain patient-specific even when both measurements are absent. Each variant produces its own otherwise matched raw-feature set, including body-size measures and clinically motivated interactions, which makes the downstream comparison an assessment of the imputation choice rather than a change of target or evaluation protocol.

## Reference dosing policies

Three non-learning policies anchor the experiments. Fixed dose assigns every patient 35 mg per week, and therefore always selects the medium arm. The IWPC clinical algorithm estimates weekly dose from age, height, weight, race, enzyme-inducing medications, and amiodarone; the pharmacogenetic algorithm augments that calculation with CYP2C9 and VKORC1 status. Their equations come directly from sections S1e and S1f of the appendix and are evaluated inside the V2 preprocessing notebook under both height/weight variants. The generated [formula-baseline report](output/preprocess_v2/warfarin_preprocess_v2_baseline_accuracy_by_hw_variant.csv) and [fixed-dose summary](results/v2_feature_set_evaluation/static_baseline_summary.csv) place fixed dosing at 61.18%, the clinical formula near 64%, and the pharmacogenetic formula near 69%.

## Contextual bandits

The principal learner is [LinUCB](src/bandits/linUCB.py), which maintains a separate linear reward model for each dose arm and combines its estimated payoff with an uncertainty bonus. `RidgeLinUCB` exposes the regularization strength of the same model, stabilizing estimation in the high-dimensional one-hot and engineered feature space. [Hybrid RidgeLinUCB](src/bandits/hybrid_linUCB.py) retains those arm-specific models while adding a shared ordinal direction for low, medium, and high dose; low- and high-dose observations can therefore inform one another without replacing their distinct coefficients or importing the pharmacogenetic formula as a feature.

The repository also examines two different responses to uncertainty and dimensionality. [LassoLinUCB](src/bandits/lasso_bandit.py) periodically fits sparse per-arm models and computes confidence bonuses on the selected active features, which is useful when many encoded covariates contribute little signal. [Linear Thompson Sampling](src/bandits/linear_thompson_sampling.py) instead samples a plausible coefficient vector from each arm’s evolving posterior and acts according to the sampled rewards, turning posterior uncertainty into randomized exploration rather than adding an explicit optimistic bonus.

## Evaluation path

The intended entry point is [preprocess_v2.ipynb](src/preprocessing/preprocess_v2.ipynb), which creates the modeling table, feature manifests, missingness reports, and static baseline measurements. [v2_feature_set_evaluation.ipynb](notebooks/v2_feature_set_evaluation.ipynb) then compares disjoint RidgeLinUCB with Hybrid RidgeLinUCB under binary and ordinal rewards, including classwise diagnostics, confusion matrices, and coefficient audits. [advanced_bandit_evaluation.ipynb](notebooks/advanced_bandit_evaluation.ipynb) applies the same cleaned feature sets and evaluation conventions to LassoLinUCB and Linear Thompson Sampling. The earlier [bandits.ipynb](notebooks/bandits.ipynb) records the project’s initial epsilon-greedy, Thompson-sampling, baseline, and LinUCB experiments, and remains useful for understanding how the modular evaluation pipeline developed.

Under the current repeated-permutation experiments, the strongest learned configuration is the hybrid ridge model at roughly 68.36% mean online accuracy. It improves substantially on assigning every patient the medium dose and exceeds the clinical formula, while remaining slightly below the pharmacogenetic reference. That gap is the present research target: the repository is an iterative experimental system for studying preprocessing, reward design, regularization, shared structure, and exploration in personalized dosing, not a clinical prescribing tool.

---

# warfarin

See the description for the warfarin project at [default_project.pdf](data/default_project.pdf). 
The warfarin dataset is at: [warfarin.csv](data/warfarin.csv).
Find more detailed details on the warfarin dataset features at [appx.pdf](data/appx.pdf).

Very useful resources I followed:\
https://banditalgs.com/2016/09/04/bandits-a-new-beginning/ \
https://banditalgs.com/2016/10/19/stochastic-linear-bandits/ \
https://www.kaggle.com/code/parsasam/reinforcement-learning-notes-multi-armed-bandits


This project idea was taken from here: https://web.stanford.edu/class/archive/cs/cs234/cs234.1224/CS234Win2019/default_project/index.html

I recently found out that warfarin came from the warf, that is, wisconsin alumni research foundation, during 
a trivia quiz at my university, UW-Madison.
