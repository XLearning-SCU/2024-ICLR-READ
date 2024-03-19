# 📚 Test-time Adaption against Multi-modal Reliability Bias

This is the official PyTorch implementation for [Test-time Adaption against Multi-modal Reliability Bias](https://openreview.net/pdf?id=TPZRq4FALB) (ICLR 2024).

## Introduction
We reveal a new challenge for multi-modal test-time adaptation dubbed modality reliability bias.

**(a)**: Imaging an autonomous vehicle equipped with camera and audio sensors driving into a foggy highway or noisy crossroad, either the visual or audio modalities would be corrupted, leading to **domain shifts** in certain modalities.<br>
**(b)**: Some corrupted modalities will lose the task-speciﬁc information and suffer from **modality reliability bias** during cross-modal fusion compared to the uncorrupted counterparts.<br>
**(c)**: The video modality contaminated with reliability bias (Video-C) has poor recognition accuracy compared to the audio modality. Both vanilla attention-based fusion (AF) and late fusion (LF) manner give inaccurate predictions compared to the single-modality ones. Instead, the proposed self-adaptive attention-based fusion (SAF) could achieve reliable fusion thus guaranteeing the performance gain in multi-modal scenarios.<br>
**(d)**: once the more informative modalities are corrupted (e.g. video for action recognition), it would be challenged to give accurate predictions. Consequently, the entropy of multi-modal predictions would be unstable. In other words, the ratio of conﬁdent predictions would decrease while the noise might dominate the predictions.

<img src="https://github.com/XLearning-SCU/2024-ICLR-READ/blob/main/figs/observation.png"  width="760" height="428" />


READ consists of two modules to handle the modality reliability bias challenge.
On the one hand, READ employs a new TTA paradigm that **modulates the attention between modalities in a self-adaptive way**, supporting reliable fusion against reliability bias. 
On the other hand, READ adopts a new objective function for robust multi-modal adaptation, where the contributions of conﬁdent predictions could be ampliﬁed and the negative impacts of noisy predictions could be mitigated.

<img src="https://github.com/XLearning-SCU/2024-ICLR-READ/blob/main/figs/framework.png"  width="760" height="268" />

## Main Requirements

torch==1.13.1 <br>
torchaudio==0.13.1<br>
timm==0.6.5<br>
scikit-learn==0.22.1<br>
numpy==1.21.6

## Benchmarks
We provide two benchmarks upon the VGGSound and Kinetics datasets for the challenge. You can download the benchmarks via [Google Cloud](https://drive.google.com/drive/folders/1SWkNwTqI08xbNJgz-YU2TwWHPn5Q4z5b?usp=sharing) or [Baidu Clound](https://pan.baidu.com/s/1Xo3IxQyd_fkzMVofDWKYVw?pwd=fnha) and use them according to the following steps. Notably, to make it easier, it is recommended to only modify the default root_path.

**Step 1. Introduce corruptions for either video or audio modality**

Modify the ```path``` and ```save_path```, and specify the ```corruption``` and ```severity``` to introduce the corruptions with different severity levels.

```
# Video-corruption:
python ./make_corruptions/make_c_video.py --corruption 'gaussian_noise' --severity 5 --data-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --save_path 'data_path/Kinetics50/image_mulframe_val256_k-C'

# Audio-corruption:
python ./make_corruptions/make_c_audio.py --corruption 'gaussian_noise' --severity 5 --data_path 'data_path/Kinetics50/audio_val256_k=50' --save_path 'data_path/Kinetics50/audio_val256_k=50-C' --weather_path 'data_path/weather_audios/'
```
**Step 2. Create JSON files for benchmarks**
JSON files are needed to reproduce our code. To make it easier, we provide our JSON files as references and you can custom your own JSON files by using the following scripts.

```
# JSON file for non-corrupted data (**Mandatory**):
python ./data_process/create_clean_json.py --refer-path 'code_path/json_csv_files/ks50_test_refer.json' --video-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --audio-path 'data_path/Kinetics50/audio_val256_k=50' --save_path 'code_path/json_csv_files/ks50'

# JSON file for video-corrupted data:
python ./data_process/create_video_c_json.py --clean-path 'code_path/json_csv_files/ks50/clean/severity_0.json' --video-c-path 'data_path/Kinetics50/image_mulframe_val256_k=50-C' --audio-path 'data_path/Kinetics50/audio_val256_k=50' --corruption 'gaussian_noise'

# JSON file for audio-corrupted data:
python ./data_process/create_audio_c_json.py --clean-path 'code_path/json_csv_files/ks50/clean/severity_0.json' --video-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --audio-c-path 'data_path/Kinetics50/audio_val256_k=50-C' --corruption 'gaussian_noise'
```

## Reproduce

You can reproduce READ by using the following script.

```
python run_read.py --dataset 'ks50' --json-root 'code_path/json_csv_files/ks50' --label-csv 'code_path/json_csv_files/class_labels_indices_ks50.csv' --pretrain_path 'code_path/pretrained_model/cav_mae_ks50.pth' --tta-method 'READ' --severity-start 5 --severity-end 5 --corruption-modality 'video'
```

- `json-root`: path of the generated jsons.
- `label-csv`: path of label csv files for class name transform.
- `pretrain_path`: the source model to be adapted. The models for VGGSound and KS50 could be accessed from [source models 1](https://www.dropbox.com/s/dl/f4wrbxv2unewss9/vgg_65.5.pth) and [source models 2](https://drive.google.com/file/d/1m38uCAfwL--RP6rWtOvGee4i2SfAzbjl/view?usp=sharing), respectively.
- `tta-method`: which TTA methods to be used. Currently, we only support `READ` and hope to release more codes for another methods soon.
- `corruption-modality`: which modality to be corrupted.
- `severity-start` & `severity-end`: which severity level of corruption you hope to be used.   

## Citation

If READ is useful for your research, please cite the following paper:
```
@inproceedings{yang2023test,
  title={Test-time adaption against multi-modal reliability bias},
  author={Yang, Mouxing and Li, Yunfan and Zhang, Changqing and Hu, Peng and Peng, Xi},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [CAV-MAE](https://github.com/YuanGongND/cav-mae?tab=readme-ov-file#pretrained-models) and [Tent](https://github.com/DequanWang/tent) licensed under Apache 2.0.
