import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, SMOTENC
NUM=["amount","transaction_hour","device_trust_score","velocity_last_24h","cardholder_age"]
BIN=["foreign_transaction","location_mismatch"]; CAT=["merchant_category"]
OUT=["amount","velocity_last_24h","device_trust_score","cardholder_age"]
df=pd.read_csv(sys.argv[1]); X=df.drop(columns=["transaction_id","is_fraud"]); y=df.is_fraud
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
tr=Xtr.assign(is_fraud=ytr); legit=tr[tr.is_fraud==0]; m=pd.Series(True,index=legit.index)
for c in OUT:
    q1,q3=legit[c].quantile([.25,.75]); i=q3-q1; m&=legit[c].between(q1-3*i,q3+3*i)
clean=pd.concat([legit[m],tr[tr.is_fraud==1]]); Xc,yc=clean.drop(columns="is_fraud"),clean.is_fraud
def pre(): return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
    ("bin",SimpleImputer(strategy="most_frequent"),BIN),("cat",OneHotEncoder(handle_unknown="ignore"),CAT)])
def evalp(name,pipe,beta=3):
    cvp=cross_val_predict(pipe,Xc,yc,method="predict_proba",cv=StratifiedKFold(5,shuffle=True,random_state=42))[:,1]
    p,r,t=precision_recall_curve(yc,cvp); fb=(1+beta**2)*p*r/(beta**2*p+r+1e-12); th=t[np.argmax(fb[:-1])]
    pipe.fit(Xc,yc); pr=pipe.predict_proba(Xte)[:,1]; pd_=pr>=th
    print(f"{name:32s} th={th:.3f} TP={int((pd_&(yte==1)).sum())}/30 FP={int((pd_&(yte==0)).sum()):3d} R={recall_score(yte,pd_):.3f} P={precision_score(yte,pd_):.3f} F1={f1_score(yte,pd_):.3f} AUC={roc_auc_score(yte,pr):.3f} PR-AUC={average_precision_score(yte,pr):.3f}",flush=True)
LR=lambda w=None: LogisticRegression(C=1.0,class_weight=w,max_iter=1000)
evalp("current: weight 10",Pipeline([("pre",pre()),("m",LR({0:1,1:10}))]))
evalp("weight 66 (balanced)",Pipeline([("pre",pre()),("m",LR("balanced"))]))
for ratio in [0.1,0.3,0.5,1.0]:
    evalp(f"SMOTE ratio={ratio}",Pipeline([("pre",pre()),("sm",SMOTE(sampling_strategy=ratio,random_state=42)),("m",LR())]))
# SMOTENC: handles categorical/binary columns properly (raw columns, before encoding)
cat_idx=[list(Xc.columns).index(c) for c in BIN+CAT]
for ratio in [0.3,1.0]:
    evalp(f"SMOTENC ratio={ratio}",Pipeline([("sm",SMOTENC(categorical_features=cat_idx,sampling_strategy=ratio,random_state=42)),("pre",pre()),("m",LR())]))
evalp("SMOTE 0.3 + weight 10",Pipeline([("pre",pre()),("sm",SMOTE(sampling_strategy=0.3,random_state=42)),("m",LR({0:1,1:10}))]))
