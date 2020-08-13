# COVID19 CT Scan Segmentation

[![made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![build status](https://img.shields.io/badge/build-passing-green.svg)]()

[![version](https://img.shields.io/badge/tensorflow-v1.15.0-yellow.svg)](https://github.com/tensorflow/tensorflow/releases)
[![build status](https://img.shields.io/badge/opencv-v4.2.0.34-red.svg)](https://pypi.org/project/opencv-python/)
[![version](https://img.shields.io/badge/nibabel-v2.3.2-cyan.svg)](https://nipy.org/nibabel/)
[![version](https://img.shields.io/badge/keras-2.3.1-green.svg)](https://pypi.org/project/Keras/)



## GOAL: Identifying infections in CT scan images of COVID19 patients using CNNs

COVID-19 patients usually develop pneumonia which rapidly progress to respiratory failure. Computed Tomography (CT) scan images play a supporative role in rapid diagnosis and the severity of the disease. Models that can find evidence of COVID-19 and/or characterize its findings can play a crucial role in optimizing diagnosis and treatment, especially in areas with a shortage of expert radiologists. This repository contains a package to identify lungs infections in CT scan images. 

For problem description and accessing raw dataset, please check [this](https://www.kaggle.com/andrewmvd/covid19-ct-scans)

For the model and results, please check [here](https://chuckyee.github.io/cardiac-segmentation/).

![Sample CT Scan Segmented](https://github.com/Mahmood-Hoseini/COVID19-CT-Scan-Segmentation/blob/master/outputs/ct-scan_sample-images.png)

## Installation

The package is written in Python and can be installed using ```bash python setup.py install``` or ```bash pip install .``` The you should be able to use the packaage:

```python
from ctseg import patient

patient_data = patient.PatientData("testing-set/patient00")
```
find out more details see ```how2useit/how2use-patient.py```.

Model is a convolutional neural network with one input (CT images) and two outputs (segmented lungs and infections) (see ```ctseg/models/convnet.py ```). Model is trained and the outputs, including weights, are saved in the ```outputs``` folder. To explore the model performance see ```scripts/evaluate.py```

To test the model on new patient data, fill out `testdir` in the `defaults.config` file and run

```bash
python -u scripts/run_on_test_patients.py defaults.config
```


## About this dataset
This dataset contains images from 20 patients (divided into 19 training and 1 testing) diagnosed with COVID-19 (see references). Files containing corresponding segmentations of lungs and infections made by experts are included as well.

## Acknowledgements

[1] - Paiva, O., 2020. CORONACASES.ORG - Helping Radiologists To Help People In More Than 100 Countries! Coronavirus Cases. See [here](https://coronacases.org/)

[2] - Glick, Y., 2020. Viewing Playlist: COVID-19 Pneumonia. Radiopaedia.Org. Available [here](https://radiopaedia.org/playlists/25887)

[3] - Kaggle dataset available [here](https://www.kaggle.com/andrewmvd/covid19-ct-scans)

