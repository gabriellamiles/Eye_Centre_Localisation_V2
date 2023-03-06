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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="http://malvikasharan.github.io/"><img src="https://avatars.githubusercontent.com/u/5370471?v=4?s=100" width="100px;" alt="Malvika Sharan"/><br /><sub><b>Malvika Sharan</b></sub></a><br /><a href="#ideas-malvikasharan" title="Ideas, Planning, & Feedback">🤔</a> <a href="#content-malvikasharan" title="Content">🖋</a></td>
      <td align="center"><a href="https://github.com/EKaroune"><img src="https://avatars.githubusercontent.com/u/58147174?v=4?s=100" width="100px;" alt="Emma Karoune"/><br /><sub><b>Emma Karoune</b></sub></a><br /><a href="#ideas-EKaroune" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/reproducible-project-template/commits?author=EKaroune" title="Documentation">📖</a></td>
      <td align="center"><a href="http://www.aleesteele.com"><img src="https://avatars.githubusercontent.com/u/18509789?v=4?s=100" width="100px;" alt="Anne Lee Steele"/><br /><sub><b>Anne Lee Steele</b></sub></a><br /><a href="#ideas-aleesteele" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/reproducible-project-template/commits?author=aleesteele" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/vhellon"><img src="https://avatars.githubusercontent.com/u/93144591?v=4?s=100" width="100px;" alt="Vicky Hellon"/><br /><sub><b>Vicky Hellon</b></sub></a><br /><a href="#ideas-vhellon" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/alan-turing-institute/reproducible-project-template/commits?author=vhellon" title="Documentation">📖</a></td>
      <td align="center"><a href="http://jending.com"><img src="https://avatars.githubusercontent.com/u/5104098?v=4?s=100" width="100px;" alt="Jennifer Ding"/><br /><sub><b>Jennifer Ding</b></sub></a><br /><a href="#content-dingaaling" title="Content">🖋</a> <a href="https://github.com/alan-turing-institute/reproducible-project-template/commits?author=dingaaling" title="Documentation">📖</a> <a href="#ideas-dingaaling" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="http://lappland.io"><img src="https://avatars.githubusercontent.com/u/51458?v=4?s=100" width="100px;" alt="Hilmar Lapp"/><br /><sub><b>Hilmar Lapp</b></sub></a><br /><a href="https://github.com/alan-turing-institute/reproducible-project-template/issues?q=author%3Ahlapp" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
# Eye_Centre_Localisation_V2
