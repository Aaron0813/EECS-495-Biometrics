
import glob as gb
import json
import PythonSDK
from PythonSDK.facepp import API, File





def build_gallery_probe(gallery_imgs, probe_imgs):
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
                faceResStr = faceResStr + "," + faceList[index]["face_token"]
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

    json.dump(gallery, open('gallery.json', 'w'))
    json.dump(probe, open('probe.json', 'w'))

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

    json.dump(genuine_scores, open('genuine_scores.json', 'w'))
    json.dump(imposter_scores, open('imposter_scores.json', 'w'))







