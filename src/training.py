"""Split -> training-only scaling/SMOTE -> fit -> untouched-test F1."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import KEYS, environment, feature_path, initialize, log_error, save_json, sha256
from .features import DIMENSIONS, extract, specification


@dataclass
class TrainConfig:
    # [PAPER] test split ratio .33, RBF, Adam, BCE+sigmoid, learning rate .01.
    # Everything else below: NOT SPECIFIED IN PAPER; implementation decisions.
    seed: int = 42
    test_size: float = 0.33
    smote_k_neighbors: int = 5
    scaling: str = "standard"
    svm_c: float = 1.0
    svm_gamma: str = "scale"
    learning_rate: float = 0.01
    hidden_size: int = 32
    layers: int = 1
    dropout: float = 0.0
    batch_size: int = 32
    epochs: int = 20
    device: str = "auto"
    sequence_layout: str = "single_step"


def counts(y):
    keys, values = np.unique(y, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(keys, values)}


def read_features(root, target, feature, dataset="sep28k"):
    path = feature_path(root, target, feature, dataset)
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Run extraction first: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["dataset"] != dataset or metadata["stutter_type"] != target:
        raise ValueError("Feature provenance does not match requested experiment")
    if metadata["feature_config"] != specification(feature):
        raise ValueError("Feature configuration changed; re-extract before training")
    if metadata["feature_csv_sha256"] != sha256(path):
        raise ValueError("Feature CSV changed after extraction; re-extract to restore provenance")
    df = pd.read_csv(path, dtype={key: str for key in KEYS})
    columns = [f"feature_{i}" for i in range(1,DIMENSIONS[feature]+1)]
    if list(df.columns) != KEYS + ["label"] + columns or df.duplicated(KEYS).any():
        raise ValueError("Feature columns or clip identities are invalid")
    X = df[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all() or not df.label.isin([0,1]).all():
        raise ValueError("Non-finite features or invalid binary labels")
    return df, X, df.label.to_numpy(dtype=np.int64), metadata


def split_and_balance(X, y, config):
    if set(np.unique(y)) != {0,1}:
        raise ValueError("Both clean and target classes must be present")
    train, test = train_test_split(np.arange(len(y)), test_size=config.test_size,
                                  stratify=y, random_state=config.seed)
    scaler = StandardScaler() if config.scaling == "standard" else None
    X_train = scaler.fit_transform(X[train]) if scaler is not None else X[train].copy()
    X_test = scaler.transform(X[test]) if scaler is not None else X[test].copy()
    before = counts(y[train])
    if min(before.values()) <= config.smote_k_neighbors:
        raise ValueError(f"SMOTE k={config.smote_k_neighbors} needs at least {config.smote_k_neighbors+1} "
                         f"training samples in each class. Observed {before}. Set --smote-k explicitly; no silent change.")
    balanced_X, balanced_y = SMOTE(random_state=config.seed,
        k_neighbors=config.smote_k_neighbors).fit_resample(X_train, y[train])
    after = counts(balanced_y)
    print(f"Training classes before SMOTE: {before}\nTraining classes after SMOTE: {after}")
    if not np.isfinite(balanced_X).all():
        raise ValueError("SMOTE produced non-finite values")
    return train, test, scaler, balanced_X, balanced_y, X_test, {"before": before, "after": after}


def make_network(dimension, classifier, config):
    import torch
    from torch import nn

    class RecurrentClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # [IMPLEMENTATION DECISION] the aggregate feature vector is ONE
            # timestep. The paper does not disclose a temporal input layout.
            self.rnn = nn.LSTM(dimension, config.hidden_size, config.layers,
                batch_first=True, bidirectional=classifier == "Bi-LSTM",
                dropout=config.dropout if config.layers > 1 else 0)
            self.output = nn.Linear(config.hidden_size * (2 if classifier == "Bi-LSTM" else 1), 1)

        def forward(self, x):
            _, (hidden, _) = self.rnn(x.unsqueeze(1))
            last = torch.cat((hidden[-2], hidden[-1]), dim=1) if classifier == "Bi-LSTM" else hidden[-1]
            return self.output(last).squeeze(1)

    return RecurrentClassifier()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_neural(X, y, classifier, config, root):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install CUDA PyTorch or use --device cpu/auto")
    device = "cuda" if config.device != "cpu" and torch.cuda.is_available() else "cpu"
    if config.device == "auto" and device == "cpu":
        print("CUDA unavailable; training on CPU.")

    def fit(on_device):
        seed_everything(config.seed)
        network = make_network(X.shape[1], classifier, config).to(on_device)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()  # mathematically sigmoid + BCE, stable evaluation
        data = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
        loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(config.seed))
        history = []
        for epoch in range(config.epochs):
            network.train()
            total = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(on_device), batch_y.to(on_device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(network(batch_X), batch_y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; no result saved")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch_y)
            history.append({"epoch": epoch+1, "training_bce": total / len(y)})
            print(f"{classifier} epoch {epoch+1}/{config.epochs}: BCE={total/len(y):.6f}", flush=True)
        return network.cpu(), history

    try:
        network, history = fit(device)
    except RuntimeError as exc:
        if device != "cuda" or config.device != "auto" or not any(
            word in str(exc).lower() for word in ("cuda", "cudnn", "cublas", "out of memory")):
            raise
        log_error(root, "training", "cuda_error", str(exc))
        print(f"CUDA failed: {exc}\nRestarting the entire run from its seed on CPU.")
        torch.cuda.empty_cache()
        device = "cpu"
        network, history = fit(device)
    state = {k: v.detach().numpy() for k, v in network.state_dict().items()}
    return state, history, device


def predict_features(bundle, X):
    import torch
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != bundle["feature_config"]["dimension"] or not np.isfinite(X).all():
        raise ValueError("Prediction input shape/non-finite error")
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)
    if bundle["classifier"] == "SVM":
        model = bundle["estimator"]
        prediction = model.predict(X)
        probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    else:
        config = TrainConfig(**bundle["train_config"])
        model = make_network(X.shape[1], bundle["classifier"], config)
        model.load_state_dict({k: torch.from_numpy(v) for k,v in bundle["state_dict"].items()})
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0,len(X),config.batch_size):
                values = torch.from_numpy(X[start:start+config.batch_size].astype(np.float32))
                parts.append(torch.sigmoid(model(values)).numpy())
        probabilities = np.concatenate(parts)
        prediction = (probabilities >= 0.5).astype(int)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite prediction probabilities")
    return prediction, probabilities


def save_evaluation(root, folder, y, predicted, probability, df, target, classifier, feature, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    score = float(f1_score(y, predicted, pos_label=1, average="binary", zero_division=0))
    report = classification_report(y, predicted, labels=[0,1], target_names=["Clean", target],
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, predicted, labels=[0,1])
    save_json(folder / "metrics.json", {"f1_score": score, "f1_average": "binary", "positive_label": 1,
                                      "classification_report": report, "dataset": dataset})
    (folder / "classification_report.txt").write_text(classification_report(y, predicted,
        labels=[0,1], target_names=["Clean",target], zero_division=0), encoding="utf-8")
    pd.DataFrame(matrix, index=["true_clean","true_target"], columns=["pred_clean","pred_target"]).to_csv(folder / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(matrix, cmap="Blues")
    for (i,j), value in np.ndenumerate(matrix):
        ax.text(j,i,str(value),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Clean",target],yticklabels=["Clean",target],
           xlabel="Predicted",ylabel="Actual",title=f"{classifier} | F1 = {score:.4f}")
    fig.tight_layout();fig.savefig(folder / "confusion_matrix.png",dpi=140);plt.close(fig)
    predictions = df[KEYS].copy()
    predictions["label"] = y
    predictions["prediction"] = predicted
    predictions["target_probability"] = probability
    predictions.to_csv(folder / "predictions.csv",index=False)
    result = {"stutter_type": target, "classifier": classifier, "feature_type": feature, "f1_score": score}
    master = root / "results" / ("results.csv" if dataset == "sep28k" else "fluencybank_results.csv")
    pd.DataFrame([result]).to_csv(master,mode="a",header=not master.exists(),index=False)
    print(f"Untouched {dataset} evaluation: F1={score:.6f}")
    return result


def train(root, target="block", feature="CQT-LTAS", classifier="SVM", config=None):
    initialize(root)
    config = config or TrainConfig()
    if config.epochs < 1 or config.batch_size < 1 or config.hidden_size < 1 or config.layers < 1:
        raise ValueError("Epochs, batch size, hidden size and layers must be positive")
    if not 0 <= config.dropout < 1 or (config.layers == 1 and config.dropout != 0):
        raise ValueError("Dropout must be in [0,1) and needs at least 2 recurrent layers")
    df, X, y, metadata = read_features(root,target,feature,"sep28k")
    train_idx, test_idx, scaler, balanced_X, balanced_y, test_X, smote = split_and_balance(X,y,config)
    del test_X  # prediction uses the saved scaler through the common inference path
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_{target}_{feature}_{classifier}"
    folder = root / "results" / run_id
    folder.mkdir(parents=True)
    model_dir = root / "models" / run_id
    model_dir.mkdir(parents=True)
    split = df[KEYS+["label"]].copy()
    split["split"] = "train"
    split.loc[test_idx,"split"] = "test"
    split.to_csv(folder / "split.csv",index=False)
    bundle = {"classifier": classifier, "feature_type": feature, "stutter_type": target,
              "feature_config": specification(feature), "train_config": asdict(config), "scaler": scaler,
              "training_dataset": "sep28k", "training_clip_keys": df.iloc[train_idx][KEYS].values.tolist()}
    history = []
    if classifier == "SVM":
        model = SVC(kernel="rbf", C=config.svm_c, gamma=config.svm_gamma,
                    probability=True, random_state=config.seed)
        model.fit(balanced_X,balanced_y)
        bundle["estimator"] = model
        device = "cpu"
    else:
        state,history,device = train_neural(balanced_X,balanced_y,classifier,config,root)
        bundle["state_dict"] = state
    prediction, probability = predict_features(bundle,X[test_idx])
    model_path = model_dir / "model.joblib"
    joblib.dump(bundle, model_path)
    # Round-trip verification exercises the exact saved artifact.
    loaded = joblib.load(model_path)
    re_prediction, re_probability = predict_features(loaded,X[test_idx[:2]])
    if not np.array_equal(re_prediction,prediction[:2]) or not np.allclose(re_probability,probability[:2]):
        raise RuntimeError("Saved model round-trip changed predictions")
    result = save_evaluation(root,folder,y[test_idx],prediction,probability,df.iloc[test_idx],
                             target,classifier,feature,"sep28k")
    manifest = {"run_id": run_id, "status": "completed", "result": result, "model": str(model_path),
                "environment": environment(), "train_config": asdict(config), "actual_device": device,
                "feature_extraction": metadata, "dataset_counts": {"all": counts(y),
                 "train": counts(y[train_idx]), "test": counts(y[test_idx])}, "smote": smote,
                "order": ["stratified split", "fit scaler on original training only",
                          "SMOTE training only", "train", "evaluate untouched test"],
                "limitations": "METHODOLOGY.md; clip split is not speaker/episode-disjoint",
                "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    save_json(folder / "configuration.json", manifest)
    save_json(model_dir / "configuration.json", manifest)
    if history:
        import matplotlib.pyplot as plt
        pd.DataFrame(history).to_csv(folder / "training_history.csv",index=False)
        fig,ax=plt.subplots();ax.plot([h["epoch"] for h in history],[h["training_bce"] for h in history])
        ax.set(xlabel="Epoch",ylabel="Training BCE",title=classifier)
        fig.tight_layout();fig.savefig(folder / "training_curve.png",dpi=140);plt.close(fig)
    config_path = root / "results/experiment_config.json"
    overall = json.loads(config_path.read_text()) if config_path.exists() else {"runs": []}
    overall.setdefault("runs",[]).append({"run_id": run_id,"configuration": str(folder / "configuration.json")})
    overall["status"] = "experiments_run"
    overall["latest_run"] = manifest
    save_json(config_path,overall)
    print(f"Saved model: {model_path}")
    return model_path, manifest


def load_bundle(path):
    bundle = joblib.load(path)  # Only load models produced by this project that you trust.
    if bundle["feature_config"] != specification(bundle["feature_type"]):
        raise ValueError("Model feature configuration differs from current extractor; use the original project version")
    return bundle


def evaluate_fluencybank(root, model_path):
    bundle = load_bundle(model_path)
    target, feature = bundle["stutter_type"], bundle["feature_type"]
    df,X,y,metadata = read_features(root,target,feature,"fluencybank")
    train_keys = {tuple(map(str,k)) for k in bundle["training_clip_keys"]}
    if train_keys.intersection(map(tuple,df[KEYS].values.tolist())):
        raise ValueError("FluencyBank clip IDs overlap training IDs; independent evaluation refused")
    if set(np.unique(y)) != {0,1}:
        raise ValueError("Independent binary evaluation needs both clean and target clips")
    predicted,probability = predict_features(bundle,X)  # no fit, no SMOTE, no refit scaler
    folder=root / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_fluencybank")
    folder.mkdir(parents=True)
    result=save_evaluation(root,folder,y,predicted,probability,df,target,bundle["classifier"],feature,"fluencybank")
    save_json(folder / "configuration.json", {"model": str(model_path), "model_sha256": sha256(Path(model_path)),
          "training_dataset": "sep28k", "evaluation_dataset": "fluencybank", "feature_extraction": metadata,
          "environment": environment(), "refit": False, "smote": False, "result": result})
    return result


def predict_audio(model_path, audio_path):
    bundle = load_bundle(model_path)
    vector, meta = extract(Path(audio_path),bundle["feature_type"])
    prediction, probability = predict_features(bundle,vector[None,:])
    label=int(prediction[0]);p=float(probability[0])
    return {"prediction": "Clean" if label == 0 else bundle["stutter_type"].title(),
            "target_probability": p, "prediction_probability": p if label else 1-p,
            "probability_method": "SVM internal training-only Platt calibration" if bundle["classifier"]=="SVM" else "sigmoid",
            "feature_dimension": len(vector), "audio": meta}
