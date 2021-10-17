import numpy as np
import matplotlib.pyplot as plt
import pickle

########################
######画第一条线##########
########################
# load feature file
pkl_file = open('p_svm.pkl', 'rb')
data_roc = pickle.load(pkl_file)
pkl_file.close()

fpr = data_roc['fpr-micro']
tpr = data_roc['tpr-micro']
micro_auc = data_roc['micro-auc']

# 绘制所有类别平均的roc曲线
plt.figure()
lw = 2
# plt.plot(fpr, tpr,
#          label='[11],[12],[34]',
#          color='deeppink', linestyle='-')
plt.plot(fpr, tpr, label='SVM', color='b', linestyle='-', linewidth=lw, markersize=12)

########################
######画第二条线##########
########################

# load feature file
pkl_file = open('p_xgb.pkl', 'rb')
data_roc = pickle.load(pkl_file)
pkl_file.close()

fpr = data_roc['fpr-micro']
tpr = data_roc['tpr-micro']
micro_auc = data_roc['micro-auc']

# plt.plot(fpr, tpr,
#          label='[12],[34]',
#          color='aqua', linestyle='--')
plt.plot(fpr, tpr, label='XGBoost', color='g', linestyle='--', linewidth=lw, markersize=12)
########################
######画第三条线##########
########################

# load feature file
pkl_file = open('p_rf.pkl', 'rb')
data_roc = pickle.load(pkl_file)
pkl_file.close()

fpr = data_roc['fpr-micro']
tpr = data_roc['tpr-micro']
micro_auc = data_roc['micro-auc']

# plt.plot(fpr, tpr,
#          label='[11],[34]',
#          color='navy', linestyle='-.')
plt.plot(fpr, tpr, label='random forest', color='r', linestyle='-.', linewidth=lw, markersize=12)

# ########################
# ######画第四条线##########
# ########################
#
# # load feature file
# pkl_file = open('0825_145441_deep.pkl', 'rb')
# data_roc = pickle.load(pkl_file)
# pkl_file.close()
#
# fpr = data_roc['fpr-micro']
# tpr = data_roc['tpr-micro']
# micro_auc = data_roc['micro-auc']
#
# # plt.plot(fpr, tpr,
# #          label='[11],[12]',
# #          color='darkorange', linestyle=':')
# plt.plot(fpr, tpr, label='[5][17]', color='c', linestyle=':', linewidth=lw, markersize=12)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC', fontdict={'family': 'Times New Roman', 'size': 16})
plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'size': 16})
plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'size': 16})
plt.yticks(fontproperties='Times New Roman', size=14)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.legend(loc="lower right")
plt.savefig('train_set.pdf')
plt.show()
