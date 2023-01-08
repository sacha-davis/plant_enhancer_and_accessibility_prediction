# contains functions that are common between both the frequency and sequence training scripts

import tensorflow as tf

from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, roc_auc_score
from scipy.stats import spearmanr  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html


def Spearman(y_true, y_pred): 
  '''
  Compute spearman coefficient that's compatible with monitoring training
  From: https://stackoverflow.com/questions/53404301/how-to-compute-spearman-correlation-in-tensorflow
  '''
  return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), 
                    tf.cast(y_true, tf.float32)], Tout = tf.float32))


def real_value_evaluation(dir_path, target_cols, saved_model, X_train, X_val, X_test, y_train, y_val, y_test):
  '''
  Given a model that predicts real-valued target(s) and data with ground truth
  Computes and saves evaluation on model+data
  '''

  train_output = saved_model.predict(X_train)  # rows, # targets
  val_output = saved_model.predict(X_val)
  test_output = saved_model.predict(X_test)

  for i, name in enumerate(target_cols):
    # grab predictions for single target
    train_pred = y_train[:,i]
    val_pred = y_val[:,i]
    test_results = y_test[:,i]

    with open(dir_path+"/results_"+name+".csv", "w") as f:
      f.write(",train,val,test\n")
      f.write("r2,"+str(r2_score(y_train[:,i], train_output[:,i]))+","+str(r2_score(y_val[:,i], val_output[:,i]))+","+str(r2_score(y_test[:,i], test_output[:,i]))+"\n")
      f.write("spearman,"+str(spearmanr(y_train[:,i], train_output[:,i])[0])+","+str(spearmanr(y_val[:,i], val_output[:,i])[0])+","+str(spearmanr(y_test[:,i], test_output[:,i])[0])+"\n")


def binary_evaluation(dir_path, target_cols, saved_model, X_train, X_val, X_test, y_train, y_val, y_test):
  '''
  Given a model that predicts real-valued target(s) and data with ground truth
  Computes and saves evaluation on model+data
  '''

  train_prob = saved_model.predict(X_train)
  val_prob = saved_model.predict(X_val)
  test_prob = saved_model.predict(X_test)

  for i, name in enumerate(target_cols):  # for every target we want to predict 
    # get class predictions for single target
    train_pred = (train_prob[:,i] > 0.5).astype("int32")
    val_pred = (val_prob[:,i] > 0.5).astype("int32")
    test_pred = (test_prob[:,i] > 0.5).astype("int32")

    # generate confusion matrix-based metrics
    train_results = get_results(y_train[:,i], train_pred)
    val_results = get_results(y_val[:,i], val_pred)
    test_results = get_results(y_test[:,i], test_pred)

    with open(dir_path+"/results_"+name+".csv", "w") as f:
      f.write(",train,val,test\n")
      f.write("accuracy,"+str(accuracy_score(y_train[:,i], train_pred))+","+str(accuracy_score(y_val[:,i], val_pred))+","+str(accuracy_score(y_test[:,i], test_pred))+"\n")
      f.write("AUC,"+str(roc_auc_score(y_train[:,i], train_prob[:,i]))+","+str(roc_auc_score(y_val[:,i], val_prob[:,i]))+","+str(roc_auc_score(y_test[:,i], test_prob[:,i]))+"\n")
      f.write("precision,"+str(train_results["precision"])+","+str(val_results["precision"])+","+str(test_results["precision"])+"\n")
      f.write("recall-sensitivity,"+str(train_results["recall-sensitivity"])+","+str(val_results["recall-sensitivity"])+","+str(test_results["recall-sensitivity"])+"\n")
      f.write("specificity,"+str(train_results["specificity"])+","+str(val_results["specificity"])+","+str(test_results["specificity"])+"\n")
      f.write("TN,"+str(train_results["TN"])+","+str(val_results["TN"])+","+str(test_results["TN"])+"\n")
      f.write("FN,"+str(train_results["FN"])+","+str(val_results["FN"])+","+str(test_results["FN"])+"\n")
      f.write("TP,"+str(train_results["TP"])+","+str(val_results["TP"])+","+str(test_results["TP"])+"\n")
      f.write("FP,"+str(train_results["FP"])+","+str(val_results["FP"])+","+str(test_results["FP"]))


def get_results(y_true, y_pred):  # positive = accessible
  '''
  Given y_true and y_pred
  Computes confusion matrix values, recall/sensitivity, specificity, and precision
  '''

  results = {}
  results["TN"], results["FP"], results["FN"], results["TP"] = confusion_matrix(y_true, y_pred).ravel()  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

  # Sensitivity, hit rate, recall, or true positive rate
  results["recall-sensitivity"] = results["TP"]/(results["TP"]+results["FN"])
  # Specificity or true negative rate
  results["specificity"] = results["TN"]/(results["TN"]+results["FP"]) 
  # Precision or positive predictive value
  results["precision"] = results["TP"]/(results["TP"]+results["FP"])

  return results

  