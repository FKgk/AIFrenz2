# [위성관측 데이터 활용 강수량 산출 AI 경진대회](https://dacon.io/competitions/official/235591/overview/)

## submit 
### lightbgm_all_31_800_2.33.csv
- public score: 2.33526 (131위)
- private score: 2.44881 (89위)


---
---


# Experiment

## Ridge(alpha=10.0) dl_train.npy shpae (1600 * 15, 1)
- fscore : 0.17027079680498733 	 maeOverFscore : 10.658128098636725



## Ridge alpha 12.5 SelectK 8
- fscore : 0.25464065823329085 	 maeOverFscore : 6.752701127612951
- fscore : 0.25458534451407944 	 maeOverFscore : 6.7528914220204515
- fscore : 0.2545837754161434 	 maeOverFscore : 6.702930965930638
- fscore : 0.25427823328070515 	 maeOverFscore : 6.747219104960767

## Ridge alpha 10.0 SelectK 8
- fscore : 0.25464065823329085 	 maeOverFscore : 6.752701129278926
- fscore : 0.25458534451407944 	 maeOverFscore : 6.75289142477857
- fscore : 0.2545837754161434 	 maeOverFscore : 6.70293096792134
- fscore : 0.25427826739578746 	 maeOverFscore : 6.747218201741558

## Ridge alpha 10.0 SelectK 8 StandardScaler
- fscore : 0.2546402018844707 	 maeOverFscore : 6.752709268799811
- fscore : 0.2545845515075978 	 maeOverFscore : 6.752907726649149
- fscore : 0.2545843704403055 	 maeOverFscore : 6.702911044140701
- fscore : 0.25427924623684933 	 maeOverFscore : 6.7471879931893675

## Ridge alpha 10.0 SelectK 8 RobustScaler
- fscore : 0.2749365314732829 	 maeOverFscore : 6.261598197378067
- fscore : 0.2752106860272466 	 maeOverFscore : 6.2545566924636615
- fscore : 0.27483218889350364 	 maeOverFscore : 6.215133163802377
- fscore : 0.2744506712189898 	 maeOverFscore : 6.258121761417609

## Ridge alpha 10.0 SelectK 8 PCA 2 dim
- fscore : 0.24107474262012732 	 maeOverFscore : 8.224389019645066
- fscore : 0.2410970325797045 	 maeOverFscore : 8.222939301354975
- fscore : 0.24133791359125045 	 maeOverFscore : 8.159670908854318
- fscore : 0.2408981513939592 	 maeOverFscore : 8.21528441818803

## Ridge alpha 10.0 SelectK 8 PCA 1 dim
- 0.16343634206703112 12.186417405981281
- 0.16353559251460106 12.177879886692459
- 0.16345101377437446 12.102961373810988
- 0.16322487488861562 12.179870422653915


## Lightbgm  
clf = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=31, \
						max_depth=- 1, learning_rate=0.01, \
                        n_estimators=400, subsample_for_bin=200000, objective=None, class_weight=None, \
                        min_split_gain=0.0, min_child_weight=0.001, \
                        min_child_samples=20, subsample=1.0, \
                        subsample_freq=0, colsample_bytree=1.0, \
                        reg_alpha=0.0, reg_lambda=0.0, random_state=None, \
                        n_jobs=- 1, silent=True, importance_type='split')

clf.fit(X[train_idx, :], Y[train_idx, 0], \
		eval_set=[(X[val_idx, :], Y[val_idx, 0])], \
        early_stopping_rounds=100, \
        verbose=True)


Did not meet early stopping. Best iteration is:
[400]	valid_0's l2: 2.45579
fscore        : 0.6035547983912278	
maeOverFscore : 1.7261039250898074

Did not meet early stopping. Best iteration is:
[400]	valid_0's l2: 2.11919
fscore        : 0.5702102319429913	
maeOverFscore : 1.7191791666443437

Did not meet early stopping. Best iteration is:
[400]	valid_0's l2: 2.43247
fscore        : 0.5894452330389313	
maeOverFscore : 1.7323487886539732

Did not meet early stopping. Best iteration is:
[400]	valid_0's l2: 2.85161
fscore        : 0.6389779388818206	
maeOverFscore : 1.8017740685707464

Did not meet early stopping. Best iteration is:
[1000]  valid_0's l2: 2.41853
fscore        : 0.6242383296335617
maeOverFscore : 1.7076647742636357

### train vs val (:112000000, 112000000:)
[400]	valid_0's l2: 1.72269
fscore        : 0.6392624073411757	
maeOverFscore : 1.7811428259684727

### 2000, 28
Did not meet early stopping. Best iteration is:
[2000]  valid_0's l2: 2.39741
fscore        : 0.6335426349159089
maeOverFscore : 1.6396066001397651


## dart
[600]   valid_0's l2: 2.58721
fscore        : 0.6066729223041292      
maeOverFscore : 1.965713243941674


## Catboost 500 0.01 4 20 Bernoulli MAE
bestTest = 0.1255453984
bestIteration = 499


