"""Beginner-friendly command line. Run python -m src.pipeline --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, TARGETS, environment, initialize, log_error, save_json
from .features import DIMENSIONS, extract_dataset, specification, validate_clip
from .references import compare, write_paper_results


def setup(root):
    from dataclasses import asdict
    from .training import TrainConfig
    initialize(root)
    write_paper_results(root)
    config=root / "results/experiment_config.json"
    if not config.exists():
        save_json(config,{"status":"not_run", "reason":"Real SEP-28k data must be processed before any research results exist",
                          "environment":environment(),"feature_defaults":specification(),
                          "train_defaults":asdict(TrainConfig()),"dataset_counts":"PENDING actual dataset audit","runs":[]})
    print(f"Project root: {root}\nPaper references written; measured results are created only by actual runs.")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT,help="Default: folder containing src (D:\\stutter-detection after extraction)")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("setup",help="Create folders, paper result table and initial run manifest")
    for name in ("prepare","extract","run"):
        sp=sub.add_parser(name)
        sp.add_argument("--target",choices=[*TARGETS,"all"],default="block")
        if name != "run":
            sp.add_argument("--dataset",choices=["sep28k","fluencybank"],default="sep28k")
        if name in ("prepare","run"):
            sp.add_argument("--labels",type=Path)
            sp.add_argument("--wav-dir",type=Path)
            sp.add_argument("--wav-template",default="{Show}{EpId}{ClipId}.wav")
        if name in ("extract","run"):
            sp.add_argument("--feature",choices=[*DIMENSIONS,"all"],default="CQT-LTAS")
        if name=="run":
            add_training(sp,all_default=True)
    sp=sub.add_parser("validate-clip")
    sp.add_argument("audio",type=Path);sp.add_argument("--feature",choices=DIMENSIONS,default="CQT-LTAS")
    sp=sub.add_parser("train")
    sp.add_argument("--target",choices=[*TARGETS,"all"],default="block")
    sp.add_argument("--feature",choices=[*DIMENSIONS,"all"],default="CQT-LTAS")
    add_training(sp)
    sp=sub.add_parser("evaluate",help="Evaluate a saved SEP-28k model on FluencyBank; never retrain")
    sp.add_argument("--model",required=True,type=Path)
    sp=sub.add_parser("predict");sp.add_argument("audio",type=Path);sp.add_argument("--model",required=True,type=Path)
    sp=sub.add_parser("demo");sp.add_argument("--model",required=True,type=Path);sp.add_argument("--port",type=int,default=8501)
    sp=sub.add_parser("compare");sp.add_argument("--dataset",choices=["sep28k","fluencybank"],default="sep28k")
    sub.add_parser("selftest",help="Isolated test fixtures; never treated as SEP-28k results")
    sub.add_parser("validate",help="Audit the actual project; missing dataset is reported as pending, never passed")
    return p


def add_training(p,all_default=False):
    p.add_argument("--classifier",choices=["SVM","LSTM","Bi-LSTM","all"],default="all" if all_default else "SVM")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--smote-k",type=int,default=5)
    p.add_argument("--scaling",choices=["standard","none"],default="standard")
    p.add_argument("--svm-c",type=float,default=1.0)
    p.add_argument("--svm-gamma",default="scale",help="scale, auto, or a positive float")
    p.add_argument("--hidden-size",type=int,default=32)
    p.add_argument("--layers",type=int,default=1)
    p.add_argument("--dropout",type=float,default=0.0)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")


def main(argv=None):
    args=parser().parse_args(argv)
    root=args.root.resolve()
    initialize(root)
    try:
        if args.command=="setup":
            setup(root);return 0
        if args.command=="validate-clip":
            validate_clip(args.audio,args.feature);return 0
        if args.command in ("selftest","validate"):
            from .validation import selftest,validate_project
            return selftest(root) if args.command=="selftest" else validate_project(root)
        if args.command=="compare":
            compare(root,args.dataset);return 0
        if args.command in ("predict","evaluate","demo"):
            from .training import predict_audio,evaluate_fluencybank
            if args.command=="predict":
                print(json.dumps(predict_audio(args.model,args.audio),indent=2))
            elif args.command=="evaluate":
                evaluate_fluencybank(root,args.model)
            else:
                from .demo import serve
                serve(root,args.model,args.port)
            return 0
        targets=list(TARGETS) if args.target=="all" else [args.target]
        dataset=getattr(args,"dataset","sep28k")
        if args.command in ("prepare","run"):
            from .dataset import prepare
            prepare(root,dataset,targets,args.labels,args.wav_dir,args.wav_template)
        if args.command=="prepare":
            return 0
        features=list(DIMENSIONS) if args.feature=="all" else [args.feature]
        if args.command in ("run","train"):
            from .training import TrainConfig,train
            classifiers=["SVM","LSTM","Bi-LSTM"] if args.classifier=="all" else [args.classifier]
            gamma=args.svm_gamma if args.svm_gamma in ("scale","auto") else float(args.svm_gamma)
            config=TrainConfig(seed=args.seed,smote_k_neighbors=args.smote_k,scaling=args.scaling,
                svm_c=args.svm_c,svm_gamma=gamma,hidden_size=args.hidden_size,layers=args.layers,
                dropout=args.dropout,batch_size=args.batch_size,epochs=args.epochs,device=args.device)
        for target in targets:
            for feature in features:
                if args.command in ("run","extract"):
                    extract_dataset(root,target,feature,dataset)
                if args.command in ("run","train"):
                    for classifier in classifiers:
                        train(root,target,feature,classifier,config)
        if args.command in ("run","train"):
            compare(root)
        return 0
    except (ValueError,FileNotFoundError,RuntimeError,OSError,ImportError) as exc:
        log_error(root,getattr(args,"audio",args.command),type(exc).__name__,str(exc))
        print(f"ERROR: {exc}")
        return 1


if __name__=="__main__":
    raise SystemExit(main())
