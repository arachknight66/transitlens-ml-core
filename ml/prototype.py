"""Restricted four-class prototype training and inference.

This module never writes the production active-model pointer. It exists to make
the best honest use of incomplete Phase 2 evidence for demonstrations.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, platform

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from .calibration import TemperatureScaler, expected_calibration_error, multiclass_brier
from .contracts import PHYSICAL_CLASSES
from .preprocessing import FrozenPreprocessor
from .uncertainty import ReviewPolicy, normalized_entropy, route_review

def _load(source: Path, split: str, feature_order: list[str], class_order):
    name={"train":"phase2_features_train.parquet","validation":"phase2_features_validation.parquet","test":"phase2_features_test.parquet"}[split]
    frame=pd.read_parquet(source/name)
    meta=pd.read_parquet(source/"phase2_feature_metadata.parquet")
    frame=frame.merge(meta[["observation_id","canonical_label"]],on="observation_id",how="left",validate="one_to_one")
    frame=frame[frame.canonical_label.isin(class_order)].copy()
    label_to_index={name:i for i,name in enumerate(class_order)}
    y=frame.canonical_label.map(label_to_index)
    if y.isna().any(): raise ValueError(f"invalid physical label in {split}")
    return frame, y.astype(int).to_numpy()

def _candidates(seed: int, num_classes: int):
    candidates={
      "extra_trees_balanced":ExtraTreesClassifier(n_estimators=700,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",n_jobs=-1,random_state=seed),
      "extra_trees_deep":ExtraTreesClassifier(n_estimators=700,min_samples_leaf=1,max_features=.7,class_weight="balanced",n_jobs=-1,random_state=seed),
      "random_forest_balanced":RandomForestClassifier(n_estimators=700,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=seed),
      "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=350,learning_rate=.06,max_leaf_nodes=31,l2_regularization=1.0,class_weight="balanced",random_state=seed),
      "logistic_regression":LogisticRegression(max_iter=3000,C=.5,class_weight="balanced",random_state=seed),
    }
    try:
        from xgboost import XGBClassifier
        candidates["xgboost_depth4"]=XGBClassifier(n_estimators=600,max_depth=4,learning_rate=.04,
            subsample=.85,colsample_bytree=.8,min_child_weight=2,reg_lambda=2,objective="multi:softprob",
            num_class=num_classes,eval_metric="mlogloss",n_jobs=-1,random_state=seed)
        candidates["xgboost_depth6"]=XGBClassifier(n_estimators=500,max_depth=6,learning_rate=.04,
            subsample=.85,colsample_bytree=.8,min_child_weight=3,reg_lambda=3,objective="multi:softprob",
            num_class=num_classes,eval_metric="mlogloss",n_jobs=-1,random_state=seed)
    except ImportError:
        pass
    return candidates

def _metrics(y, probs, class_order):
    pred=probs.argmax(1); report=classification_report(y,pred,labels=range(len(class_order)),target_names=class_order,output_dict=True,zero_division=0)
    return {"targets":len(y),"macro_f1":float(f1_score(y,pred,average="macro")),
            "ece":expected_calibration_error(y,probs),"brier_score":multiclass_brier(y,probs),
            "per_class":{name:report[name] for name in class_order}}

def _selection_score(metrics):
    pc=metrics["per_class"]
    return (.45*metrics["macro_f1"] + .20*pc["exoplanet_transit"]["f1-score"] +
            .20*pc["eclipsing_binary"]["recall"] + .15*pc["blend_contamination"]["f1-score"])

def _tune_review(y, probs):
    entropy=normalized_entropy(probs); best=None
    for minimum in np.arange(.40,.81,.05):
      for maximum_entropy in np.arange(.65,1.01,.05):
        accepted=(probs.max(1)>=minimum)&(entropy<=maximum_entropy); coverage=accepted.mean()
        if coverage<.50 or not accepted.any(): continue
        score=f1_score(y[accepted],probs[accepted].argmax(1),average="macro",zero_division=0)
        candidate=(score,coverage,float(minimum),float(maximum_entropy))
        if best is None or candidate[:2]>best[:2]: best=candidate
    if best is None: return ReviewPolicy()
    return ReviewPolicy(minimum_probability=best[2],maximum_normalized_entropy=best[3])

def train_prototype(source: Path, output: Path, seed: int=42, class_order=PHYSICAL_CLASSES) -> dict:
    source=source.resolve(); output=output.resolve(); output.mkdir(parents=True,exist_ok=True)
    feature_order=json.loads((source/"phase2_feature_order.json").read_text())
    class_order=tuple(class_order)
    train,y_train=_load(source,"train",feature_order,class_order); val,y_val=_load(source,"validation",feature_order,class_order); test,y_test=_load(source,"test",feature_order,class_order)
    preprocessor=FrozenPreprocessor(feature_order).fit(train)
    X_train=preprocessor.transform(train); X_val=preprocessor.transform(val); X_test=preprocessor.transform(test)
    trials=[]; fitted={}
    for name,model in _candidates(seed,len(class_order)).items():
        model.fit(X_train,y_train); probs=model.predict_proba(X_val); metrics=_metrics(y_val,probs,class_order)
        trials.append({"model":name,"selection_score":_selection_score(metrics),**metrics}); fitted[name]=model
    trials.sort(key=lambda row:row["selection_score"],reverse=True); selected=trials[0]["model"]; model=fitted[selected]
    val_raw=model.predict_proba(X_val); calibrator=TemperatureScaler().fit(np.log(np.clip(val_raw,1e-15,1)),y_val)
    val_probs=calibrator.transform(np.log(np.clip(val_raw,1e-15,1)))
    policy=_tune_review(y_val,val_probs)
    test_raw=model.predict_proba(X_test); test_probs=calibrator.transform(np.log(np.clip(test_raw,1e-15,1)))
    review=route_review(test_probs,policy,ood=preprocessor.out_of_range(test))
    test_metrics=_metrics(y_test,test_probs,class_order); test_metrics["review_rate"]=float(np.mean([bool(x) for x in review]))
    test_metrics["coverage"]=1-test_metrics["review_rate"]
    joblib.dump(model,output/"model.joblib"); joblib.dump(preprocessor,output/"preprocessor.joblib"); joblib.dump(calibrator,output/"calibration.joblib")
    (output/"feature_order.json").write_text(json.dumps(feature_order,indent=2))
    policy_json={"minimum_probability":policy.minimum_probability,"maximum_normalized_entropy":policy.maximum_normalized_entropy,
                 "maximum_disagreement":policy.maximum_disagreement}
    (output/"review_policy.json").write_text(json.dumps(policy_json,indent=2))
    pd.DataFrame([{k:v for k,v in row.items() if k not in {"per_class"}} for row in trials]).to_csv(output/"model_comparison.csv",index=False)
    prediction=pd.DataFrame({"observation_id":test.observation_id,"canonical_label":test.canonical_label,
                             "predicted_class":[class_order[i] for i in test_probs.argmax(1)],"review_required":[bool(x) for x in review]})
    for i,name in enumerate(class_order): prediction[f"prob_{name}"]=test_probs[:,i]
    prediction.to_parquet(output/"blind_test_predictions.parquet",index=False)
    hashes={name:_sha(output/name) for name in ["model.joblib","preprocessor.joblib","calibration.joblib","feature_order.json","review_policy.json"]}
    record={"status":"DEVELOPMENT_RESTRICTED","production_eligible":False,"ml_enabled_for_prototype":True,
            "model_id":f"prototype-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}","selected_model":selected,
            "class_order":list(class_order),"source":str(source),"seed":seed,"python":platform.python_version(),
            "validation_metrics":_metrics(y_val,val_probs,class_order),"blind_test_metrics":test_metrics,"review_policy":policy_json,
            "trials":trials,"artifact_hashes":hashes,"limitations":["partial TPF coverage","no Gaia cache","not production eligible","photometry does not confirm planets"]}
    (output/"evaluation_record.json").write_text(json.dumps(record,indent=2))
    (output/"prototype_model_card.md").write_text(_card(record))
    return record

def _sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()

def _card(record):
    m=record["blind_test_metrics"]
    return f"""# TransitLens restricted prototype model

- Status: **DEVELOPMENT_RESTRICTED**
- Production eligible: **false**
- Model: {record['selected_model']}
- Physical classes: {', '.join(record['class_order'])}
- Blind targets: {m['targets']}
- Blind macro-F1: {m['macro_f1']:.4f}
- ECE: {m['ece']:.4f}
- Brier score: {m['brier_score']:.4f}
- Review rate: {m['review_rate']:.4f}

This prototype ranks astrophysical interpretations; it does not confirm planets. Missing spatial evidence and distribution shift route cases to review.
"""

class PrototypeModel:
    def __init__(self,path: Path):
        self.path=path; self.record=json.loads((path/"evaluation_record.json").read_text())
        for name,expected in self.record["artifact_hashes"].items():
            if _sha(path/name)!=expected: raise RuntimeError(f"prototype artifact hash mismatch: {name}")
        self.model=joblib.load(path/"model.joblib"); self.preprocessor=joblib.load(path/"preprocessor.joblib"); self.calibrator=joblib.load(path/"calibration.joblib")
        self.order=json.loads((path/"feature_order.json").read_text()); self.class_order=tuple(self.record["class_order"]); self.policy=ReviewPolicy(**json.loads((path/"review_policy.json").read_text()))
    def predict(self,frame: pd.DataFrame):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Skipping features without any observed values")
            X=self.preprocessor.transform(frame)
        raw=self.model.predict_proba(X); probs=self.calibrator.transform(np.log(np.clip(raw,1e-15,1)))
        reasons=route_review(probs,self.policy,ood=self.preprocessor.out_of_range(frame)); output=[]
        for p,why in zip(probs,reasons):
            physical=self.class_order[int(p.argmax())]; output.append({"predicted_astrophysical_class":physical,
              "routing_outcome":"review_required" if why else physical,"probabilities":dict(zip(self.class_order,map(float,p))),
              "review_required":bool(why),"review_reasons":why,"model_id":self.record["model_id"],"model_status":"development_restricted"})
        return output
