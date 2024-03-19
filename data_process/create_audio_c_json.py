import os
import json
import copy
import random
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--clean-path', type=str, default='/xlearning/mouxing/workspace/TTA/READ/_code_clean/json_csv_files/vgg/clean/severity_0.json')
parser.add_argument('--video-path', type=str, default="/xlearning/mouxing/dataset/ImageAudio/VGGSound/image_mulframe_test")
parser.add_argument('--audio-c-path', type=str, default="/xlearning/mouxing/dataset/ImageAudio/VGGSound/audio_test-C")
parser.add_argument('--corruption', nargs='*', default=['all'])
args = parser.parse_args()

with open(args.clean_path, 'r') as f:
    data = json.load(f)

dic_list = data['data']

tmp_dic_list = copy.deepcopy(dic_list)

severity_list = range(1, 6)
if args.corruption[0] == 'all':
    corruption_list = [
    'gaussian_noise',
    'crowd',
    'rain',
    'thunder',
    'traffic',
    'wind'
    ]
else:
    corruption_list = args.corruption

mixed_corruption_severity_list = []
for corruption in corruption_list:
    mixed_severity_list = []
    for severity in severity_list:
        save_path = os.path.join(os.path.dirname(args.clean_path)[:-5], 'audio')

        if not os.path.exists(os.path.join(save_path, corruption)):
            os.makedirs(os.path.join(save_path, corruption))
        dic_list = []
        for dic in tmp_dic_list:
            new_dic = {
                "video_id": dic.get("video_id"), # + '-{}-{}'.format(method, severity),
                "wav": os.path.join(args.audio_c_path, corruption, 'severity_{}'.format(severity), '{}.wav'.format(dic.get("video_id"))),
                "video_path": args.video_path,
                "labels": dic.get("labels")
            }
            dic_list.append(new_dic)
        print(len(dic_list))
        random.shuffle(dic_list)
        new_json = {"data": dic_list}
        with open(os.path.join(save_path, corruption, 'severity_{}.json'.format(severity)), "w") as file1:
            json.dump(new_json, file1, indent=1)