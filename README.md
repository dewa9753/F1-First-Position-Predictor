# F1 Podium Predictor (Gradient Boosting)

Small project that predicts whether an F1 driver will finish on the podium (top 3) in a race using a gradient boosting classifier and race-related features (qualifying times, grid position, circuit, team/driver history).

## Current Status
Project code is being finalized. Then, the project will not be worked on any further.

## Key ideas
- Problem: binary classification — podium (yes/no).
- Model: gradient boosting (sklearn.ensemble)
- Feature examples: q3 qualifying lap time, grid position, team/driver form, circuit characteristics, and others that are known before race start.

## Data
- Source: [Kaggle F1 Dataset](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data/)
- Preprocessing: cleaned missing qualifying/finish values, engineered certain features like best previous lap times using the existing data.

## Quickstart
Common command chain:
1. py .\process_data.py --force-final
2. py .\EDA.py --show-plots
3. py .\train_test_model.py --force-train

## Results & Analysis
- Evaluation metrics: Accuracy and F1 score
- 79% accuracy, 79% F1 in predicting if a given driver on a specific circuit with other features will finish on the podium.
- A random guess has a probability of 50% accuracy, so the model provides 29% more accuracy than a random guess.

## Further Development Ideas
- Split chronologically by race/season to avoid leakage.
- Use per-circuit embeddings or target encoding for circuit/driver to capture track effects.
- Get more data from other datasets about circuit-specific characteristics

## License
MIT
