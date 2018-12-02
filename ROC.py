import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

# y_test = [0, 1, 0, 1, 1, 0]
# y_score = [24.56, 97.34, 34.65, 91.98, 89.98, 53.87]


def draw_roc(y_test, y_score):
    print("y_test",y_test)
    print("y_score", y_score)
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.pause(3)
    # plt.savefig('ROC.jpg')
