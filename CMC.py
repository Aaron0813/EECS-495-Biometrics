import numpy as np
import matplotlib.pyplot as plt

''' 
# test data
predict_label = np.array([
    [97.32, 23.56, 34.98],
    [18.21, 91.22, 45.98],
    [44.33, 29.87, 89.77]
])

test_y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
'''
# CMC curve
def draw_cmc(predict_label, test_y):
    test_cmc = []  # store accuracy
    sort_index = np.argsort(-predict_label,axis=1)
    print("sort_index",sort_index)

    actual_index = np.argmax(test_y,1)

    print("actual_index \n", actual_index)
    predict_index = np.argmax(predict_label,1)
    temp = np.cast['float32'](np.equal(actual_index,predict_index))
    test_cmc.append(np.mean(temp))

    for i in range(sort_index.shape[1]-1):
        for j in range(len(temp)):
            if temp[j]==0:
                predict_index[j] = sort_index[j][i+1]
        temp = np.cast['float32'](np.equal(actual_index,predict_index))
        test_cmc.append(np.mean(temp))


    plt.figure()

    print(test_cmc)

    x = np.arange(0, sort_index.shape[1])
    plt.plot(x, test_cmc, color="red", linewidth=2, label="CMC")
    plt.xlabel("Rank")
    plt.ylabel("Matching Rate")
    plt.legend()
    plt.title("CMC Curve")
    plt.savefig('CMC.jpg')