# NASA shuttle telemetry anomaly detection

I built this as a Streamlit diagnostic panel around an unsupervised anomaly detector for
NASA shuttle radiator telemetry. The detector is trained on the OpenML `shuttle` dataset
(nine sensor channels, about 56,000 rows, seven classes collapsed to normal against
anomaly) and then served from three pre-trained scikit-learn pipelines.

The models are the detection half. The lifecycle around them, including data versioning,
sweeps, a model registry, drift monitoring and retraining, lives in
[shuttle-nasa-mlops](https://github.com/S4afou/shuttle-nasa-mlops).

## What is here

```text
app.py                                   Streamlit panel: sensor inputs, prediction, history
train_mlops.py                           training entry point, logs metrics and the model to W&B
shuttle_lof_pipeline.pkl                 Local Outlier Factor pipeline
best_anomaly_detection_pipeline.pkl      Isolation Forest pipeline
shuttle_ocsvm_99f1.pkl                   One-Class SVM pipeline
requirements.txt
.devcontainer/devcontainer.json
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens on `http://localhost:8501`. Pick a detector in the sidebar, enter the eight
radiator sensor readings (A2 to A9), and run the diagnostic. The panel reports the verdict,
the decision score and a short history of the session.

Weights and Biases is optional. The app tries to log each prediction to the
`nasa-shuttle-mlops` project so the same charts as the monitoring script receive traffic,
and falls back silently when it cannot authenticate or the machine is offline.

## Models

Three pipelines are selectable so the same inputs can be compared across detectors:

- **Local Outlier Factor.** Density based, sensitive to the local neighbourhood.
- **Isolation Forest.** Random splits, cheap and robust on mixed scales.
- **One-Class SVM.** Boundary around the normal class.

All three sit behind a `StandardScaler` inside a scikit-learn `Pipeline`, so the raw sensor
values are accepted as-is.

The best registered sweep reached an anomaly F1 of 0.979 on its hold-out split. That number
and the drift experiment are documented in the MLOps repository; this repository is the
inference side.

## Why the pickles are committed

The three pipelines are about 33 MB in total. I committed them on purpose: the app is a
demo, and a clone that runs without a training job is worth more than a tidy tree. If you
would rather retrain, `train_mlops.py` fetches the data, trains the model and registers the
artifact.

## Dataset

- Source: OpenML, `shuttle` (version 1), the NASA shuttle radiator position dataset
- Shape: nine numeric sensor channels, about 56,000 rows
- Labels: seven original classes, mapped to `0` for normal and `1` for anomaly
- Access: public, fetched with `sklearn.datasets.fetch_openml`
- Citation: NASA, distributed through OpenML

The class balance is heavily skewed toward normal operation, which is why the project
reports F1 rather than accuracy.

## Licence

Course and portfolio work. No licence is granted for reuse of the dataset, which follows
the OpenML terms.

More about this project on my portfolio:
https://safouane-medjahri.vercel.app/projects/shuttle-anomaly-detection
