# Eye Centre Localisation

*A The Turing Way inspired project to enable reproducibility in data science.*

## About this Repository

This repository is for the detection of eye regions in images. The data the model is trained on is
all very similar to the image shown below (and as such is very constrained):

## Repo Structure

Inspired by [Cookie Cutter Data Science](https://github.com/drivendata/cookiecutter-data-science)

```
├── LICENSE
├── README.md          <- The top-level README for users of this project.
├── CODE_OF_CONDUCT.md <- Guidelines for users and contributors of the project.
├── CONTRIBUTING.md    <- Information on how to contribute to the project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project.
│   │
│   ├── data           <- Scripts to download or generate data
│   │   ├── build_dataset.py
│   │   ├── combine_labels_from_different_sources.py
│   │   ├── combine_left_and_right.py
│   │   ├── create_dataset.py
│   │   └── create_eye_centre_training_dataset.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── config.py
│   │   ├── ECL_predictor.py
│   │   ├── improved_ECL.py
│   │   ├── model_utils.py
│   │   ├── predict_model.py
│   │   ├── predict_model_right.py
│   │   ├── run_prediction.sh
│   │   ├── run_predictions_right.sh
│   │   ├── train_on_incorrect_results.py
│   │   └── train_model.py
│   │
│   └── visualisation  <- Scripts to create exploratory and results oriented visualisations
│       ├── create_videos_from_prediction.py
│       ├── create_videos_of_labels.py
│       ├── inspect_combined_labels.py
│       ├── save_images_with_predicted_centres.py
│       ├── predictions
│       ├── prediction_videos
│       └── labelInspectionVideos
└──
```
