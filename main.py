from pprint import pformat
import numpy as np
import glob as gb
import json
# import PythonSDK
from PythonSDK.facepp import API, File
import ROC
import CMC
import seaborn as sns
from scipy.stats import norm
import copy
import PythonSDK.ImagePro

gallery_imgs = './FaceSet_gallery/*.jpg'       # build faceSet
probe_imgs = './FaceSet_probe/*.jpg'

# gallery info
# Key: Face Token
# Value: Person id
gallery = {}
# probe info
# Key: Face Token
# Value: Person id
probe = {}


# init api
api = API()

'''  '''

# delete useless face_set
api.faceset.delete(outer_id='face_recognition', check_empty=0)
# # 1.create faceSet
ret = api.faceset.create(outer_id='face_recognition')
#
# # 2.add pics to faceSet(face_token)

img_path = gb.glob(gallery_imgs)
for img in img_path:
    faceResStr = ""
    res = api.detect(image_file=File(img))
    # get person's name
    person_name = img[-12:][0:5]
    print(person_name)
    print(res)
    faceList = res["faces"]
    for index in range(len(faceList)):
        gallery[faceList[index]["face_token"]] = person_name
        if index == 0:
            faceResStr = faceResStr + faceList[index]["face_token"]
        else:
            faceResStr = faceResStr + ","+faceList[index]["face_token"]
    api.faceset.addface(outer_id='face_recognition', face_tokens=faceResStr)

print(gallery)

# build probe set
img_path = gb.glob(probe_imgs)
for img in img_path:
    faceResStr = ""
    res = api.detect(image_file=File(img))
    # get person's name
    person_name = img[-12:][0:5]
    print(person_name)
    print(res)
    faceList = res["faces"]
    probe[faceList[0]["face_token"]] = person_name

print("probe")
print(probe)

json.dump(gallery, open('gallery.json','w'))
json.dump(probe, open('probe.json','w'))


'''   '''
with open("./gallery.json", 'r') as g:
    gallery = json.load(g)

with open("./probe.json", 'r') as p:
    probe = json.load(p)

genuine_scores = []
imposter_scores = []

for gallery_key, gallery_value in gallery.items():
    for probe_key, probe_value in probe.items():
        result = api.compare(face_token1=gallery_key, face_token2=probe_key)
        res = {}
        res['gallery'] = gallery_value
        res['probe'] = probe_value
        res['confidence'] = result['confidence']
        if gallery_value == probe_value:
            genuine_scores.append(res)
        else:
            imposter_scores.append(res)

json.dump(genuine_scores, open('genuine_scores.json','w'))
json.dump(imposter_scores, open('imposter_scores.json','w'))






########################################3
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



