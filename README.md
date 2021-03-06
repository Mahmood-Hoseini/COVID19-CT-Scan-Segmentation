# COVID19 CT Scan Segmentation

[![build status](https://img.shields.io/badge/build-passing-green.svg)]()

[![build status](https://img.shields.io/badge/made%20with-python-cyan.svg)](https://www.python.org/)
[![version](https://img.shields.io/badge/tensorflow-v1.15.0-gold.svg)](https://github.com/tensorflow/tensorflow/releases)
[![build status](https://img.shields.io/badge/opencv-v4.2.0.34-gold.svg)](https://pypi.org/project/opencv-python/)
[![version](https://img.shields.io/badge/nibabel-v2.3.2-gold.svg)](https://nipy.org/nibabel/)
[![version](https://img.shields.io/badge/keras-2.3.1-gold.svg)](https://pypi.org/project/Keras/)



## GOAL: Identifying infections in CT scan images of COVID19 patients using CNNs

COVID-19 patients usually develop pneumonia which rapidly progress to respiratory failure. Computed Tomography (CT) scan images play a supporative role in rapid diagnosis and the severity of the disease. Models that can find evidence of COVID-19 and/or characterize its findings can play a crucial role in optimizing diagnosis and treatment, especially in areas with a shortage of expert radiologists. This repository contains a package to identify lungs infections in CT scan images. 

![Sample CT Scan](https://github.com/Mahmood-Hoseini/COVID19-CT-Scan-Segmentation/blob/master/outputs/gif-pid11-cts.gif)

To download training and testing datasets see [this](https://drive.google.com/drive/folders/1Y_LtHDZBq0K-B8zrkkN3SsAmAw6M7TJi?usp=sharing)

![Sample CT Scan Segmented](https://github.com/Mahmood-Hoseini/COVID19-CT-Scan-Segmentation/blob/master/outputs/ct-scan_sample-images.png)

## Installation

The package is written in Python and can be installed using ```python setup.py install``` or ```pip install .``` The you should be able to use the packaage:

```python
from ctseg import patient

patient_data = patient.PatientData("testing-set/patient00")
```
`PatientData` loads CT images which are used to train a convolutional network. Find out more details see ```how2useit/how2use-patient.py```.

<a href="Sample Segmented Lungs"><img src="https://github.com/Mahmood-Hoseini/COVID19-CT-Scan-Segmentation/blob/master/outputs/segmented-lungs.png" align="middle" height="400" ></a>

<a href="Sample Lung Mask"><img src="https://github.com/Mahmood-Hoseini/COVID19-CT-Scan-Segmentation/blob/master/outputs/lung-mask-and-bbox.png" align="middle" height="150" ></a>


Using segmented lungs, CT image were cropped and fed into a convolutional network to train segmenting for infections (see `ctseg/models/convnet.py `). Models were trained and the outputs, including weights, were saved in the `outputs` folder. To explore the model performance see `scripts/evaluate.py`

<a href="Sample Segmented Infections"><img src="https://github.com/Mahmood-Hoseini/COVID19-CT-Scan-Segmentation/blob/master/outputs/segmented-infections.png" align="middle" height="400" ></a>


To test the model on new patient data, fill out `testdir` in the `defaults.config` file and run

```bash
python -u scripts/run_on_test_patients.py defaults.config
```

<a href="Sample predicted output"><img src="https://github.com/Mahmood-Hoseini/COVID19-CT-Scan-Segmentation/blob/master/outputs/actualvs.pred-patient00-frame047.png" align="middle" height="300" ></a>


## About this dataset
This dataset contains images from 61 patients (divided into 55 training and 6 testing) diagnosed with COVID-19 (see references). Files containing corresponding segmentations of lungs and infections made by experts are included as well.

## Acknowledgements

[1] - Paiva, O., 2020. CORONACASES.ORG - Helping Radiologists To Help People In More Than 100 Countries! Coronavirus Cases. See [here](https://coronacases.org/)

[2] - Glick, Y., 2020. Viewing Playlist: COVID-19 Pneumonia. Radiopaedia.Org. Available [here](https://radiopaedia.org/playlists/25887)

[3] - Kaggle dataset available [here](https://www.kaggle.com/andrewmvd/covid19-ct-scans)

[4] - Zhang, Kang, Xiaohong Liu, Jun Shen, Zhihuan Li, Ye Sang, Xingwang Wu, Yunfei Zha et al. "Clinically applicable AI system for accurate diagnosis, quantitative measurements, and prognosis of covid-19 pneumonia using computed tomography." Cell (2020).

[5] - Cardiac MRI Segmentation [blog](https://chuckyee.github.io/cardiac-segmentation/) and [github](https://chuckyee.github.io/cardiac-segmentation/)

