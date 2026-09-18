"""Transcription of Tables I and II, never passed off as measured results."""
import pandas as pd
from .common import PAPER_DOI, TARGETS

FEATURES = ["LTAS","CQT-LTAS","GT-LTAS","SFF-LTAS","MFCC","ZTWCC"]
SEP = {
"SVM": [
[.5261,.6233,.4785,.5960,.5964], [.6331,.6624,.6315,.6885,.6660],
[.5396,.6644,.4770,.6411,.6441], [.5983,.6610,.6025,.6565,.6460],
[.5580,.6025,.5246,.6089,.6514], [.5969,.6028,.4893,.6211,.6532]],
"LSTM": [
[.6509,.6799,.6370,.5732,.6279], [.7343,.7607,.7696,.7287,.7949],
[.7555,.6143,.7928,.7622,.7862], [.7017,.7476,.7859,.7290,.6020],
[.5585,.6173,.6427,.6929,.6851], [.6593,.6337,.7099,.7061,.7175]],
"Bi-LSTM": [
[.6665,.6402,.6466,.6059,.6628], [.7516,.7644,.7760,.7905,.7789],
[.7341,.7153,.7149,.7363,.7688], [.7259,.7346,.7468,.7006,.6910],
[.7129,.7109,.7105,.6977,.7242], [.7223,.7116,.7128,.7005,.7468]]}
FLUENCY = {
"SVM": [[.6267,.6513,.6291,.6936,.6132], [.5412,.6725,.5312,.6231,.6247]],
"LSTM": [[.7242,.7523,.7532,.7065,.7637], [.7616,.6324,.7965,.7710,.7526]],
"Bi-LSTM": [[.7621,.7541,.7876,.7865,.7723], [.7218,.7232,.7287,.7442,.7587]]}


def write_paper_results(root):
    rows=[]
    for dataset, table, features in [("sep28k",SEP,FEATURES),("fluencybank",FLUENCY,["CQT-LTAS","GT-LTAS"])]:
        for classifier, scores in table.items():
            for feature, values in zip(features,scores):
                for target, score in zip(TARGETS,values):
                    rows.append({"dataset":dataset,"stutter_type":target,"classifier":classifier,
                                 "feature_type":feature,"f1_score":score,"source":"[PAPER]", "doi":PAPER_DOI})
    path=root / "results/paper_reference_results.csv"
    pd.DataFrame(rows).to_csv(path,index=False,float_format="%.4f")
    return path


def compare(root, dataset="sep28k"):
    paper_path=write_paper_results(root)
    our_path=root / "results" / ("results.csv" if dataset=="sep28k" else "fluencybank_results.csv")
    if not our_path.exists():
        raise FileNotFoundError("No measured results yet. Run training/evaluation first; paper scores are separate.")
    paper=pd.read_csv(paper_path)
    ours=pd.read_csv(our_path)
    comparison=ours.merge(paper.loc[paper.dataset.eq(dataset)],
        on=["stutter_type","classifier","feature_type"],suffixes=("_ours","_paper"),validate="many_to_one")
    comparison["difference"]=comparison.f1_score_ours-comparison.f1_score_paper
    comparison["caveat"]="Reference-informed reproduction; selection, numerical filters, input layout and F1 convention may differ"
    path=root / "results" / f"{dataset}_paper_comparison.csv"
    comparison.to_csv(path,index=False)
    print(comparison.to_string(index=False))
    return path
