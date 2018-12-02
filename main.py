import ROC
import CMC
import json
import build_gallery_probe
import seaborn as sns
from scipy.stats import norm
import copy
import PythonSDK.ImagePro
import numpy as np

gallery_imgs = './FaceSet_gallery/*.jpg'       # build faceSet
probe_imgs = './FaceSet_probe/*.jpg'

# to save time, just comment this line and get compare result from json files
# build_gallery_probe.build_gallery_probe(gallery_imgs, probe_imgs)

'''  '''
# calc ROC
genuine_scores = []
imposter_scores = []

with open("./genuine_scores.json", 'r') as load_f:
    genuine_file = json.load(load_f)

for genuine in genuine_file:
    genuine_scores.append(genuine['confidence'])


with open("./imposter_scores.json", 'r') as load_f:
    imposter_file = json.load(load_f)

for imposter in imposter_file:
    imposter_scores.append(imposter['confidence'])


# draw Genius and Imposter
sns.set(color_codes=True)
# sns.distplot(genuine_scores, fit=norm, bins= 10, hist=True, rug=True)
genuine_scores_new = copy.deepcopy(genuine_scores)
imposter_scores_new = copy.deepcopy(imposter_scores)

genuine_scores_new[:] = [((100 - x) / 100) for x in genuine_scores_new]
imposter_scores_new[:] = [((100 - x) / 100) for x in imposter_scores_new]
sns.distplot(genuine_scores_new, kde_kws={"color": "g", "lw": 2, "label": "genuine"}, bins= 5)
sns.distplot(imposter_scores_new, kde_kws={"color": "y", "lw": 2, "label": "impostor"}, bins= 5, hist=True)


'''   '''
print(genuine_scores)
print(imposter_scores)

y_test = [0 for n in range(len(genuine_scores) + len(imposter_scores))]
for n in range(len(genuine_scores)):
    y_test[n] = 1

genuine_scores_new = copy.deepcopy(genuine_scores)
imposter_scores_new = copy.deepcopy(imposter_scores)
y_score = genuine_scores_new + imposter_scores_new

ROC.draw_roc(y_test, y_score)
print(y_score)



#####################################
# calc CMC
'''
'''
array = np.zeros((len(genuine_scores), len(genuine_scores)))
print(array.shape)
length = len(genuine_scores)
index = 0
for i in range(length):

    for j in range(length):
        if i == j:
            array[i][j] = genuine_scores[i]
        else:
            # print("i = ", i, "j = ", j)
            # print(len(genuine_scores)*i + j)
            array[i][j] = imposter_scores[index]
            index += 1

print(array)
test_y = np.eye(length, M=None, k=0)

CMC.draw_cmc(array, test_y)

''' '''
