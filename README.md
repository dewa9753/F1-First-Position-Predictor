# F1 Podium Predictor (Gradient Boosting)

Small project that predicts whether an F1 driver will finish on the podium (top 3) in a race using a gradient boosting classifier and race-related features (qualifying times, grid position, circuit, team/driver history).

## Current Status
Project is completed and will not be worked on further.

## Key ideas
- Problem: binary classification — podium (yes/no).
- Model: gradient boosting (sklearn.ensemble)
- Feature examples: q3 qualifying lap time, grid position, team/driver form, circuit characteristics, and others that are known before race start.

## Data
- Source: [Kaggle F1 Dataset](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data/)
- Preprocessing: cleaned missing qualifying/finish values, engineered certain features like best previous lap times using the existing data.

## Quickstart
Common commands:
- py .\process_data.py --force-final
- py .\EDA.py --show-plots
- py .\train_test_model.py --force-train

## Results & Analysis
- Evaluation metrics: Accuracy and F1 score
- 27% accuracy, 25% F1 in predicting the final position of a given driver in any race
- the model provides 19% more accuracy than a random guess since the model only considers final positions of less than 13

## Reproducibility & Further Development
- Split chronologically by race/season to avoid leakage.
- Use per-circuit embeddings or target encoding for circuit/driver to capture track effects.
- Get more data from other datasets about circuit-specific characteristics

## License
MIT