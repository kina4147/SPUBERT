# SPU-BERT: Faster Human Multi-Trajectory Prediction from Socio-Physical Understanding of BERT.

# Codes for SPU-BERT: Faster Human Multi-Trajectory Prediction from Socio-Physical Understanding of BERT.

![$V^2$-Net](./overview.png)

## Abstract

Understanding and forecasting future trajectories of agents are critical for behavior analysis, robot navigation, autonomous cars, and other related applications.
Previous methods mostly treat trajectory prediction as time sequence generation.
Different from them, this work studies agents' trajectories in a "vertical" view, i.e., modeling and forecasting trajectories from the spectral domain.
Different frequency bands in the trajectory spectrums could hierarchically reflect agents' motion preferences at different scales.
The low-frequency and high-frequency portions could represent their coarse motion trends and fine motion variations, respectively.
Accordingly, we propose a hierarchical network V$^2$-Net, which contains two sub-networks, to hierarchically model and predict agents' trajectories with trajectory spectrums.
The coarse-level keypoints estimation sub-network first predicts the "minimal" spectrums of agents' trajectories on several "key" frequency portions.
Then the fine-level spectrum interpolation sub-network interpolates the spectrums to reconstruct the final predictions.
Experimental results display the competitiveness and superiority of V$^2$-Net on both ETH-UCY benchmark and the Stanford Drone Dataset.



## Requirements

The codes are developed with python 3.8.
Additional packages used are included in the `requirements.txt` file.
We recommend installing the above versions of the python packages in a virtual environment (like the `conda` environment), otherwise there *COULD* be other problems due to the package version conflicts.

Run the following command to install the required packages in your python  environment:

```bash
pip install -r requirements.txt
```

## Training On Your Datasets



### Dataset

ETH/UCY Datasets
Standford Drone Dataset (SDD)
Before training `V^2-Net` on your own dataset, you should add your dataset information to the `datasets` directory.
See [this document](./datasetFormat.md) for details.

## Evaluation

You can use the following command to evaluate the `V^2-Net` performance end-to-end:

```bash
python main.py \
  --model V \
  --loada A_MODEL_PATH \
  --loadb B_MODEL_PATH
```

Where `A_MODEL_PATH` and `B_MODEL_PATH` are the folders of the two sub-networks' weights.

## Pre-Trained Models

We have provided our pre-trained model weights to help you quickly evaluate the `V^2-Net` performance.
We have uploaded our model weights in the `weights` folder.
It contains model weights trained on `ETH-UCY` by the `leave-one-out` stragety, and model weights trained on `SDD` via the dataset split method from [SimAug](https://github.com/JunweiLiang/Multiverse).

Please note that we do not use dataset split files like trajectron++ or trajnet for several reasons.
For example, the frame rate problem in `ETH-eth` sub-dataset, and some of these splits only consider the `pedestrians` in the SDD dataset.
We process the original full-dataset files from these datasets with observations = 3.2 seconds (or 8 frames) and predictions = 4.8 seconds (or 12 frames) to train and test the model.
Detailed process codes are available in `./scripts/add_ethucy_datasets.py`, `./scripts/add_sdd.py`, and `./scripts/sdd_txt2csv.py`.
See deatils in [issue#1](https://github.com/cocoon2wong/Vertical/issues/1).
(Thanks [@MeiliMa](https://github.com/MeiliMa))

You can start the quick evaluation via the following commands:

```bash
for dataset in eth hotel univ zara1 zara2 sdd
  python main.py \
    --model V \
    --loada ./weights/vertical/a_${dataset} \
    --loadb ./weights/vertical/b_${dataset}
```

After the code running, you will see the output in the `./test.log` file:

Please note that the results may fluctuate slightly at each model implementation due to the random sampling in the model (which is used to generate multiple stochastic predictions).
In addition, we shrunk all SDD data by a scale factor of 100 when training the model.
The data recorded in the `./test.log` multiplied by 100 is the result we report in the paper.

You can also start testing the fast version of `V^2-Net` by passing the argument `--loadb l` like:



The `--loadb l` will replace the original stage-2 spectrum interpolation sub-network with the simple linear interpolation method.
Although it may reduce the prediction performance, the model will implement much faster.
You can see the model output in `./test.log` like:

```log
[2022-07-26 10:17:57,955][INFO] `V`: Results from ./weights/vertical/a_eth, l, eth, {'ADE(m)': 0.2517119, 'FDE(m)': 0.37815523}
...
[2022-07-26 10:18:05,915][INFO] `V`: Results from ./weights/vertical/a_hotel, l, hotel, {'ADE(m)': 0.112576276, 'FDE(m)': 0.16336456}
...
[2022-07-26 10:18:42,540][INFO] `V`: Results from ./weights/vertical/a_univ, l, univ, {'ADE(m)': 0.21333231, 'FDE(m)': 0.35480896}
...
[2022-07-26 10:23:39,660][INFO] `V`: Results from ./weights/vertical/a_zara1, l, zara1, {'ADE(m)': 0.21019873, 'FDE(m)': 0.31065288}
...
[2022-07-26 10:23:57,347][INFO] `V`: Results from ./weights/vertical/a_zara2, l, zara2, {'ADE(m)': 0.1556495, 'FDE(m)': 0.25072886}
...
[2022-07-26 10:45:53,313][INFO] `V`: Results from ./weights/vertical/a_sdd, l, sdd, {'ADE(m)': 0.06888708, 'FDE(m)': 0.106946796}
```

We have prepared model outputs that work correctly on the zara1 dataset, details of which can be found [here](https://github.com/cocoon2wong/Vertical/actions).

If you have the dataset videos and put them into the `videos` folder, you can draw the visualized results by adding the `--draw_reuslts 1` argument.
Plsase specify the video clip that you want to draw trajectories on (for example `SOME_VIDEO_CLIP`) by adding arguments `--test_mode one` and `--force_set SOME_VIDEO_CLIP`.
If you want to draw visualized trajectories like what our paper shows, you can add the additional `--draw_distribution 2` argument.
For example, you can download videos from datasets' official websets, and draw results on the `SDD-hyang2` video via the following command:

```bash
python main.py \
  --model V \
  --loada ./weights/vertical/a_sdd \
  --loadb ./weights/vertical/b_sdd \
  --draw_results 1 \
  --draw_distribution 2 \
  --test_mode one \
  --force_set hyang2
```

![Visualization](./fig_vis.png)

## Evaluation of the Usage of Spectrums

We design the minimal vertical model to directly evaluate the metrics improvements brought by the usage of DFT (i.e., the trajectory spectrums).
The minimal V model considers nothing except agents' observed trajectories when forecasting.
You can start a quick training to see how the DFT helps improve the prediction accuracy by changing the argument `--T` between `[none, fft]` via the following scripts:

```bash
for ds in eth hotel univ zara1 zara2
  for T in none fft
    python main.py \
      --model mv \
      --test_set ${ds} \
      --T ${T}
```

You can also [download](drive.google.com) (⚠️NOT UPLOAD YET) and unzip our weights into the `weights/vertical_minimal` folder, then run the following test scripts:

```bash
for name in FFTmv mv
  for ds in eth hotel univ zara1 zara2
    python main.py --load ./weights/vertical_minimal/${name}${ds}
```

Test results will be saved in the `test.log` file.
You can find the following results if everything runs correctly:

```log
[2022-07-06 10:28:59,536][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmveth, eth, {'ADE(m)': 0.79980284, 'FDE(m)': 1.5165437}
[2022-07-06 10:29:02,438][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmvhotel, hotel, {'ADE(m)': 0.22864725, 'FDE(m)': 0.38144386}
[2022-07-06 10:29:15,459][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmvuniv, univ, {'ADE(m)': 0.559813, 'FDE(m)': 1.1061481}
[2022-07-06 10:29:19,675][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmvzara1, zara1, {'ADE(m)': 0.45233154, 'FDE(m)': 0.9287788}
[2022-07-06 10:29:25,595][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmvzara2, zara2, {'ADE(m)': 0.34826145, 'FDE(m)': 0.71161735}
[2022-07-06 10:29:29,694][INFO] `MinimalV`: ./weights/vertical_minimal/mveth, eth, {'ADE(m)': 0.83624077, 'FDE(m)': 1.666721}
[2022-07-06 10:29:32,632][INFO] `MinimalV`: ./weights/vertical_minimal/mvhotel, hotel, {'ADE(m)': 0.2543166, 'FDE(m)': 0.4409294}
[2022-07-06 10:29:45,396][INFO] `MinimalV`: ./weights/vertical_minimal/mvuniv, univ, {'ADE(m)': 0.7743274, 'FDE(m)': 1.3987076}
[2022-07-06 10:29:49,126][INFO] `MinimalV`: ./weights/vertical_minimal/mvzara1, zara1, {'ADE(m)': 0.48137394, 'FDE(m)': 0.97067535}
[2022-07-06 10:29:54,872][INFO] `MinimalV`: ./weights/vertical_minimal/mvzara2, zara2, {'ADE(m)': 0.38129684, 'FDE(m)': 0.7475274}
```

You can find the considerable ADE and FDE improvements brought by the DFT (or called the trajectory spectrums) in the above logs.
Please note that the prediction performance is quite bad due to the simple structure of the *minimal* model, and it considers nothing about agents' interactions and multimodality.

## Args Used

Please specific your customized args when training or testing your model through the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 --ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages when training and testing are listed below.
Args with `argtype='static'` means that their values can not be changed once after training.

<!-- DO NOT CHANGE THIS LINE -->
### Basic args

- `--K_train`, type=`int`, argtype=`'static'`.
  Number of multiple generations when training. This arg only works for `Generative Models`.
  The default value is `10`.
- `--K`, type=`int`, argtype=`'dynamic'`.
  Number of multiple generations when test. This arg only works for `Generative Models`.
  The default value is `20`.


### Vertical args

- `--K_train`, type=`int`, argtype=`'static'`.
  Number of multiple generations when training.
  The default value is `1`.
- `--K`, type=`int`, argtype=`'dynamic'`.
  Number of multiple generations when evaluating. The number of trajectories predicted for one agent is calculated by `N = args.K * args.Kc`, where `Kc` is the number of style channels.
  The default value is `1`.

## Thanks

Codes of the Transformers used in this model comes from [TensorFlow.org](https://www.tensorflow.org/tutorials/text/transformer);  
Dataset csv files of ETH-UCY come from [SR-LSTM (CVPR2019) / E-SR-LSTM (TPAMI2020)](https://github.com/zhangpur/SR-LSTM);  
Original dataset annotation files of SDD come from [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/), and its split file comes from [SimAug (ECCV2020)](https://github.com/JunweiLiang/Multiverse);  
[@MeiliMa](https://github.com/MeiliMa) for dataset suggestions.





## Citation  
If you find this work useful, it would be grateful to cite our paper!

```bib
@article{na2022knowledge,
  title={SPU-BERT: Faster Human Multi-Trajectory Prediction from Socio-Physical Understanding of BERT},
  author={Ki-In Na, Ue-Hwan Kim, and Jong-Hwan Kim},
  journal={Knowledge Based Systems (submitted)},
  year={2022}
}
```

## Contact us

Ki-In Na ([@kina4147](https://github.com/kina4147)): kina4147@gmail.com  
