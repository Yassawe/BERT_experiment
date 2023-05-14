# TRAINING LOSS
#grep 'loss' reports/x1.report | grep learning_rate | awk '{print $2}'

# VALIDATION LOSS
# grep 'eval_loss' reports/haha_rs1_ag1.report | grep eval_accuracy | awk '{print $2}'

# TEST ACCURACY
# grep 'eval_accuracy' reports/haha_rs1_ag1.report | grep eval_loss | awk '{print $4}'

# F1 score
grep 'eval_f1' reports/x2.report | awk '{print $6}'