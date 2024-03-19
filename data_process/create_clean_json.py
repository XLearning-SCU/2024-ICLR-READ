import os
import json
import re
import argparse

def process_files(args):
    # load refer json
    json_file_path = args.refer_path
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    data = data['data']
    ytid_to_labels = {dic['video_id']: dic['labels'] for dic in data}

    # generate new json
    new_data = []
    folder_files = os.listdir(args.audio_path)

    for filename in folder_files:
        wav = os.path.join(args.audio_path, filename)
        video_path = args.video_path
        video_id = filename.replace('.wav', '')
        labels = None

        dictionary = {
            "video_id": video_id,
            "wav": wav,
            "video_path": video_path,
            "labels": labels
        }
        new_data.append(dictionary)

    # search labels in refer json
    for dictionary in new_data:
        video_id = dictionary.get('video_id')

        # if video_id of new json in ytid_to_labels of refer json, then add labels to new json:
        if video_id in ytid_to_labels:
            labels = ytid_to_labels[video_id]
            dictionary['labels'] = labels
        else:
            print(video_id)

    new_json = {"data": new_data}
    args.save_path = os.path.join(args.save_path, 'clean')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_path = os.path.join(args.save_path, 'severity_0.json')
    with open(args.save_path, "w") as file1:
        json.dump(new_json, file1, indent=1)
    print(f"success, the json files contains {len(new_data)} items.")

# 
parser = argparse.ArgumentParser()
parser.add_argument('--refer-path', type=str, default="/xlearning/mouxing/workspace/TTA/READ/_code_clean/json_csv_files/ks50_test_refer.json")
parser.add_argument('--video-path', type=str, default="/xlearning/mouxing/dataset/ImageAudio/Kinetics400/K50-clipped/image_mulframe_val256_k=50")
parser.add_argument('--audio-path', type=str, default="/xlearning/mouxing/dataset/ImageAudio/Kinetics400/K50-clipped/audio_val256_k=50")
parser.add_argument('--save-path', type=str, default="/xlearning/mouxing/workspace/TTA/READ/_code_clean/json_csv_files/ks50")
args = parser.parse_args()
process_files(args)
