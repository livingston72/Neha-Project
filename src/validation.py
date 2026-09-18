"""Automated numerical/integration tests, plus honest real-project validation."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .common import KEYS,TARGETS,environment,feature_path,initialize,save_json,stem
from .dataset import AudioError,prepare,read_audio,read_labels
from .features import DIMENSIONS,extract,extract_dataset,ltas_statistics,preprocess


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def fixture(root):
    """TEST SIGNALS ONLY. Never place these in the user's research dataset."""
    initialize(root)
    wav=root / "dataset/clips/stuttering-clips/clips";wav.mkdir(parents=True)
    rng=np.random.default_rng(719)
    rows=[]
    for i in range(102):
        row={"Show":"TEST_ONLY","EpId":"01","ClipId":str(i).zfill(4),
             **dict.fromkeys(TARGETS.values(),0),"Start":0,"Stop":0.25}
        if i>=36:
            group=0 if i<54 else 1+(i-54)//12
            row[list(TARGETS.values())[group]]=2
        sr=16000 if i%2==0 else 8000
        t=np.arange(int(sr*(0.20+0.01*(i%5))))/sr
        wave=0.2*np.sin(2*np.pi*(130+7*i)*t)*(1+0.2*np.sin(2*np.pi*8*t))
        wave+=rng.normal(0,0.005,len(t))
        path=wav/f"{row['Show']}{row['EpId']}{row['ClipId']}.wav"
        sf.write(path,wave,sr)
        rows.append(row)
    for i,kind in enumerate(["missing","empty","corrupt","short","overlap"],102):
        row={"Show":"TEST_ONLY","EpId":"01","ClipId":str(i).zfill(4),**dict.fromkeys(TARGETS.values(),0)}
        path=wav/f"{row['Show']}{row['EpId']}{row['ClipId']}.wav"
        if kind=="empty": sf.write(path,np.empty(0),16000)
        elif kind=="corrupt": path.write_bytes(b"not a wav")
        elif kind=="short": sf.write(path,np.ones(80)*0.1,8000)
        elif kind=="overlap":
            row["Block"]=2;row["Interjection"]=1;sf.write(path,np.ones(1600)*0.1,8000)
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "dataset/SEP-28k_labels.csv",index=False)
    return wav


def selftest(report_root):
    import torch
    torch.set_num_threads(1)
    from .banks import FS,cqt_design,iter_cqt,iter_sff
    from .training import TrainConfig,evaluate_fluencybank,predict_audio,split_and_balance,train
    from .demo import HTML,make_server
    from .references import compare,write_paper_results
    initialize(report_root)
    checks=[]
    transcript=io.StringIO()
    def check(name,function):
        try:
            with contextlib.redirect_stdout(transcript):
                function()
            checks.append({"check":name,"status":"PASS"})
        except Exception as exc:
            checks.append({"check":name,"status":"FAIL","error":str(exc)})
            traceback.print_exc(file=transcript)
        print(f"{checks[-1]['status']}: {name}",flush=True)

    with tempfile.TemporaryDirectory(prefix="ltas_software_tests_") as folder:
        root=Path(folder);wav=fixture(root)
        typical=root / "typical.wav"
        time=np.arange(48000)/16000
        waveform=0.3*np.sin(2*np.pi*250*time)
        sf.write(typical,np.c_[waveform,waveform/2],16000)
        state={}

        def audio_and_shapes():
            x,meta=preprocess(typical)
            require(meta["original_sample_rate"]==16000 and len(x)==24000,"3-second resampling failed")
            require(np.isclose(meta["original_duration"],3),"Incorrect duration")
            v,m=extract(typical)
            require(m["number_of_frames"]==150 and m["rms_matrix_shape"]==[107,150],"Framing/RMS mismatch")
            require(m["filtered_signal_shape"]==[106,24000] and v.shape==(1069,),"Feature/filter mismatch")
            require(not m["nan_count"] and not m["infinity_count"],"Non-finite values")
            state["typical_clip_validation"]=m
            centers,support,gain=cqt_design(24000)
            require(len(centers)==106 and centers[0]==0 and centers[-1]==4000,"Boundary filters missing")
            require(np.allclose(centers[1:-1],10*2**(np.arange(104)/12)),"CQT formula changed")
            # Frequency-selectivity check, not just output dimensions.
            energies=np.array([np.mean(abs(v)**2) for v in iter_cqt(x)])
            require(abs(centers[energies.argmax()]-250)<25,"CQT bank does not select a 250-Hz tone")
        check("16 kHz stereo -> mono 8 kHz; 106 filters; 107x150 RMS; 1069 finite features",audio_and_shapes)

        def bad_audio():
            for number,kind in [(102,"missing_audio"),(103,"empty_audio"),(104,"invalid_audio"),(105,"short_audio")]:
                try: read_audio(wav/f"TEST_ONLY01{number:04}.wav")
                except AudioError as exc: require(exc.kind==kind,f"{number}: expected {kind}, got {exc.kind}")
                else: raise AssertionError(f"{kind} was not rejected")
            nan=root/"nan.wav";sf.write(nan,np.r_[np.nan,np.zeros(1599)],8000,subtype="FLOAT")
            try: read_audio(nan)
            except AudioError as exc: require(exc.kind=="invalid_audio","NaN wrong error")
            else: raise AssertionError("NaN WAV accepted")
        check("Missing, zero-sample, corrupted, short and NaN WAVs rejected distinctly",bad_audio)

        def stats():
            r=np.array([[1.,3.],[2.,6.]])
            b=np.sqrt([5.,20.]);v=ltas_statistics(r,b)
            expected=np.r_[2,[2,4]/b[0],[np.sqrt(2),np.sqrt(8)],
                [np.sqrt(2),np.sqrt(8)]/b[0],[np.sqrt(2),np.sqrt(8)]/b,
                [0,0],[1,1],[2,4],[2,4]/b[0],[2,4]/b[0]]
            require(v.shape==(19,) and np.allclose(v,expected),"Reference formulas/redundant-feature removal mismatch")
            impulse=np.r_[1.,np.zeros(159)]
            filtered=list(iter_sff(impulse))
            pole=.98*np.exp(-2j*np.pi*1000/FS)
            require(np.allclose(filtered[50],pole**np.arange(160)),"SFF transfer-function impulse response mismatch")
        check("All ten LTAS formulas against hand values; SFF against analytic impulse response",stats)

        def edge_features():
            for name in DIMENSIONS:
                vector,_=extract(typical,name)
                require(vector.shape==(DIMENSIONS[name],) and np.isfinite(vector).all(),name)
                silence=root/"silence.wav";sf.write(silence,np.zeros(1600),8000)
                vector,_=extract(silence,name)
                require(np.isfinite(vector).all(),name+" silence failed")
            uneven=root/"uneven.wav";sf.write(uneven,np.ones(319)*.1,8000)
            _,meta=extract(uneven)
            require(meta["number_of_frames"]==1 and meta["discarded_tail_samples"]==159,"Tail silently padded")
        check("Six feature dimensions, silent signals, one frame and incomplete tails",edge_features)

        def preparation():
            summary=prepare(root,targets=list(TARGETS))
            require(summary["total_samples"]==107 and summary["overlapping_labels"]==1,"Label counts mismatch")
            for key in ["missing_audio","empty_audio","invalid_audio","short_audio"]:
                require(summary["audio_status_counts"][key]==1,"Audit failed: "+key)
            for target in TARGETS:
                valid=pd.read_csv(root/"data"/f"{stem(target)}_valid.csv")
                require(valid.label.eq(0).sum()==36,"Clean selection changed")
                require(valid.label.eq(1).sum()==(18 if target=="block" else 12),"Pure-label filtering failed")
                require(not valid.ClipId.eq(106).any(),"Overlap included as pure")
            require(len(pd.read_csv(root/"data/clean_vs_block.csv"))==58,"Original selection not preserved")
            state["fixture_dataset_counts"]=summary
        check("Audit all labels and WAVs; preserve original; pure selection for all five tasks",preparation)

        def leakage():
            X=np.arange(60*4,dtype=float).reshape(60,4);y=np.r_[np.zeros(40,int),np.ones(20,int)]
            config=TrainConfig()
            a,b,scaler,resampled,labels,test,distribution=split_and_balance(X,y,config)
            require(not set(a)&set(b),"Split overlap")
            require(np.allclose(scaler.mean_,X[a].mean(axis=0)),"Scaler fit outside original training partition")
            require(np.array_equal(test,scaler.transform(X[b])),"Test data changed")
            changed=X.copy();changed[b]+=1e9
            aa,bb,ss,rr,ll,tt,dd=split_and_balance(changed,y,config)
            require(np.array_equal(resampled,rr) and np.array_equal(labels,ll),"Test data affected SMOTE")
            require(len(set(distribution["after"].values()))==1,"SMOTE did not balance")
        check("Test-set perturbation cannot change training scaling or SMOTE; deterministic split",leakage)

        def end_to_end():
            write_paper_results(root)
            for target in TARGETS:
                output=extract_dataset(root,target)
                df=pd.read_csv(output)
                require(df.shape[1]==1073,"Feature CSV metadata/dimension mismatch")
                if target=="block":
                    classifiers=["SVM","LSTM","Bi-LSTM"]
                else: classifiers=["SVM"]
                for classifier in classifiers:
                    model,manifest=train(root,target,classifier=classifier,
                        config=TrainConfig(epochs=2,hidden_size=8,batch_size=16,device="cpu"))
                    require(model.exists() and np.isfinite(manifest["result"]["f1_score"]),"Model/results not saved")
                    predicted=predict_audio(model,typical)
                    require(predicted["feature_dimension"]==1069 and 0<=predicted["target_probability"]<=1,"Bad inference")
                    if target=="block" and classifier=="SVM": state["model"]=model
            require(len(pd.read_csv(root/"results/results.csv"))==7,"Master results missing runs")
            require(len(pd.read_csv(root/"results/paper_reference_results.csv"))==120,"Paper transcription count mismatch")
            require(compare(root).exists(),"Comparison missing")
        check("All five binary tasks; SVM/LSTM/Bi-LSTM training, F1, model reload, reports and curves",end_to_end)

        def independent():
            source=pd.read_csv(root/"dataset/SEP-28k_labels.csv",dtype={k:str for k in KEYS}).iloc[:54].copy()
            source["Show"]="TEST_EXTERNAL"
            # Different generated fixtures, never presented as speech or dataset evidence.
            for i,row in source.iterrows():
                t=np.arange(2000)/8000
                path=wav/f"{row.Show}{row.EpId}{row.ClipId}.wav"
                sf.write(path,.15*np.sin(2*np.pi*(310+3*i)*t),8000)
            source.to_csv(root/"dataset/fluencybank_labels.csv",index=False)
            prepare(root,"fluencybank",["block"]);extract_dataset(root,dataset="fluencybank")
            before=state["model"].read_bytes()
            result=evaluate_fluencybank(root,state["model"])
            require(before==state["model"].read_bytes(),"External evaluation modified model")
            require(np.isfinite(result["f1_score"]),"External evaluation failed")
        check("Independent-dataset route loads fixed SEP model and does not retrain",independent)

        def http_demo():
            require('type="file"' in HTML and 'fetch(\'/predict\'' in HTML,"Upload UI missing")
            server=make_server(root,state["model"],port=0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                base=f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(base,timeout=15) as response:
                    require(response.status==200 and b"Clean or Block?" in response.read(),"Demo page failed")
                request=urllib.request.Request(base+"/predict",data=typical.read_bytes(),headers={"Content-Type":"audio/wav"})
                with urllib.request.urlopen(request,timeout=30) as response:
                    result=json.load(response)
                direct=predict_audio(state["model"],typical)
                # Uploads deliberately have a different temporary filename.
                result["audio"].pop("audio_path")
                direct["audio"].pop("audio_path")
                require(result==direct,"Demo preprocessing/predictions differ from CLI")
                request=urllib.request.Request(base+"/predict",data=b"invalid")
                try: urllib.request.urlopen(request,timeout=15)
                except urllib.error.HTTPError as exc: require(exc.code==400,"Bad upload wrong status")
                else: raise AssertionError("Bad WAV upload accepted")
            finally:
                server.shutdown();thread.join();server.server_close()
        check("Local HTTP demo: page, WAV upload, identical inference, corrupted upload error",http_demo)

        def guards():
            file=root/"dataset/SEP-28k_labels.csv"
            original=pd.read_csv(file,dtype={k:str for k in KEYS})
            duplicate=root/"duplicate.csv";pd.concat([original,original.iloc[:1]]).to_csv(duplicate,index=False)
            try: read_labels(duplicate)
            except ValueError: pass
            else: raise AssertionError("Duplicate IDs accepted")
            invalid=original.copy();invalid.loc[0,"Block"]=-1;invalid.to_csv(duplicate,index=False)
            try: read_labels(duplicate)
            except ValueError: pass
            else: raise AssertionError("Negative votes accepted")
            from .training import read_features
            feature_file=feature_path(root,"block","CQT-LTAS")
            with feature_file.open("a") as f:f.write("\n")
            try: read_features(root,"block","CQT-LTAS")
            except ValueError: pass
            else: raise AssertionError("Edited features accepted despite hash mismatch")
        check("Duplicate/invalid labels and stale feature files cannot silently enter training",guards)

        # No fixture models, generated tones or synthetic scores escape into research folders.
        report={"scope":"SOFTWARE TESTS ONLY: generated numerical fixtures; not SEP-28k/FluencyBank research",
                "environment":environment(),"checks":checks,
                "passed":sum(c["status"]=="PASS" for c in checks),
                "failed":sum(c["status"]=="FAIL" for c in checks),
                "typical_clip_validation":state.get("typical_clip_validation"),
                "actual_dataset_evaluation":"PENDING: run on Windows with real data",
                "windows_cuda_validation":"PENDING: not available on macOS CPU host"}
        save_json(report_root/"results/software_validation.json",report)
        (report_root/"results/software_test_log.txt").write_text(
            "SOFTWARE TEST FIXTURES ONLY. All audio, counts, model fits and F1 scores below are isolated numerical tests.\n"
            "They are NOT measured SEP-28k or FluencyBank research results. Test models/audio were removed.\n\n"
            + transcript.getvalue(),encoding="utf-8")
    return 1 if report["failed"] else 0


def validate_project(root):
    """Validate real artifacts independently; missing work never receives PASS."""
    checks=[]
    def record(name,condition,detail=""):
        checks.append({"check":name,"status":"PASS" if condition else "PENDING_OR_FAILED","detail":detail})
    record("Dataset found",(root/"dataset/SEP-28k_labels.csv").is_file() and
           (root/"dataset/clips/stuttering-clips/clips").is_dir())
    record("Labels found",(root/"dataset/SEP-28k_labels.csv").is_file())
    record("WAV directory found",(root/"dataset/clips/stuttering-clips/clips").is_dir())
    summary_path=root/"results/sep28k_dataset_summary.json"
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    task=summary.get("tasks",{}).get("block",{})
    record("WAV matching and valid dataset generated",task.get("valid_clips",0)>0)
    record("Empty audio detection audit performed",bool(summary.get("audio_status_counts")))
    selected=root/"data/clean_vs_block_valid.csv"
    if selected.exists():
        frame=pd.read_csv(selected)
        record("Every selected valid WAV still exists",not frame.empty and all(Path(p).is_file() for p in frame.audio_path))
    else: record("Every selected valid WAV still exists",False)
    validation=root/"data/ltas_features.validation.json"
    shape_report=json.loads(validation.read_text()) if validation.exists() else {}
    for name,condition in [
        ("8 kHz resampling",shape_report.get("resampled_sample_rate")==8000),
        ("20 ms rectangular framing",shape_report.get("frame_size")==160),
        ("106 CQT filters and full band",shape_report.get("number_of_filters")==106 and shape_report.get("components")==107),
        ("RMS and ten LTAS measures",shape_report.get("statistics_per_component")==10),
        ("1069 features, no NaN/Infinity",shape_report.get("final_feature_shape")==[1069] and
             shape_report.get("nan_count")==0 and shape_report.get("infinity_count")==0)]: record(name,condition)
    feature_ok=False
    try:
        from .training import read_features
        df,X,y,metadata=read_features(root,"block","CQT-LTAS")
        feature_ok=len(df)>0 and metadata["failed_clips"]==0
    except (OSError,ValueError,KeyError):pass
    record("Feature CSV complete, finite and provenance verified",feature_ok)
    manifests=[]
    for p in (root/"models").glob("*/configuration.json"):
        manifests.append(json.loads(p.read_text()))
    for classifier in ["SVM","LSTM","Bi-LSTM"]:
        runs=[m for m in manifests if m.get("result",{}).get("classifier")==classifier
              and m.get("result",{}).get("stutter_type")=="block"
              and m.get("result",{}).get("feature_type")=="CQT-LTAS"]
        record(f"{classifier}: split, SMOTE, F1 and saved model",any(
            m.get("status")=="completed" and Path(m.get("model","")).is_file() for m in runs))
    record("Measured results saved",(root/"results/results.csv").is_file())
    software=root/"results/software_validation.json"
    sw=json.loads(software.read_text()) if software.exists() else {}
    record("Demo upload software test",any(c["status"]=="PASS" and c["check"].startswith("Local HTTP demo") for c in sw.get("checks",[])))
    record("README and methodology documented",(root/"README.md").is_file() and (root/"METHODOLOGY.md").is_file())
    report={"scope":"REAL PROJECT VALIDATION", "environment":environment(),"checks":checks,
            "complete":all(c["status"]=="PASS" for c in checks)}
    save_json(root/"results/project_validation.json",report)
    for c in checks:print(f"{c['status']}: {c['check']}")
    return 0 if report["complete"] else 2
