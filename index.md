---
layout: default
---

# Project Proposal

## Introduction/Background

With the rate of respiratory ailments on the rise, the stress imposed on global health becomes more prevalent [1]. These ailments, if left untreated, can leave an individual with irreversible damage to their respiratory system by impairing lung function. Recent developments in medical technology have yielded datasets of recorded lung sounds, opening up new possibilities for automated lung sound examination. 

We analyze the largest and most prominent of these datasets, ICBHI [4, 5]; this dataset consists of 920 annotated recordings (via 4 different devices) from 126 patients totaling 5.5 hours and 6898 breathing cycles, as well as structured data on each patient’s age, sex, height, weight, and respiratory disease.

## Problem Definition

Motivated by our desire to improve early detection of serious respiratory disease, we analyze audio samples using machine learning methods to differentiate between 6 different kinds of respiratory conditions (COPD, URTI, Bronchiectasis, Pneumonia, Bronchiolitis, and healthy) as a way of pre-screening patients. We believe this project will be best utilized as a web application to expedite triage in medical establishments or to enhance at-home symptom checking. While most published work on the topic has focused on identifying abnormal sounds, such as crackles and wheezes, with high levels of accuracy [1,2,6], we go further and attempt to identify the specific pulmonary disease of a patient given their demographic information and auscultation audio sample.

## Methods

To preprocess the audio data, we will convert the sound clips to mel spectrograms, which can then be passed into the models we propose below. Additionally, we plan to apply a Butterworth band-pass filter to separate the actual lung sounds from background signals, such as heartbeat audio [2]. To ensure standardized data, we will trim each audio sample so that each sample has the same number of breathing cycles. We will also employ noise-injection and speed-modulating data augmentation methods to generate more samples [6]. 

For our supervised learning models, we will experiment with 3 primary methods: (1) various neural network architectures using convolutional and recurrent layers (TensorFlow), (2) an audio spectrogram transformer [7] (Hugging Face), and a (3) random forest (scikit-learn) to model the mix of categorical and continuous variables in our dataset. We will also experiment with ensembles and chains of these models, such as weighted sums of predictions and feeding the input of the neural net into the random forest. For our unsupervised learning method, we plan on clustering the tabular demographic data using a Gaussian mixture model to identify populations of high-risk groups for each disease.

## (Potential Results and Discussion)

Based on our analysis of related work, we aim to achieve a 60% accuracy rate in correctly classifying respiratory diseases and an F1 score of 0.7. In a medical context, limiting false positives and false negatives is imperative, necessitating the emphasis of F1 score over accuracy to specifically account for both the precision and recall of our model. We will also evaluate our deep neural nets in-training using a class-balanced focal adjustment [3] to cross entropy loss to address the dataset’s imbalance in disease representation.

## References

[1] S. Bae et al., “Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification,” in INTERSPEECH 2023, ISCA, Aug. 2023, pp. 5436–5440. doi: 10.21437/Interspeech.2023-1426.

[2] S. Gairola, F. Tom, N. Kwatra, and M. Jain, “RespireNet: A Deep Neural Network for Accurately Detecting Abnormal Lung Sounds in Limited Data Setting,” in 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Nov. 2021, pp. 527–530. doi: 10.1109/EMBC46164.2021.9630091.

[3] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-Balanced Loss Based on Effective Number of Samples,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2019, pp. 9260–9269. doi: 10.1109/CVPR.2019.00949.

[4] “Respiratory Sound Database.” Accessed: Feb. 22, 2024. [Online]. Available: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database

[5] B. M. Rocha et al., “Α Respiratory Sound Database for the Development of Automated Classification,” in Precision Medicine Powered by pHealth and Connected Health, vol. 66, N. Maglaveras, I. Chouvarda, and P. De Carvalho, Eds., Singapore: Springer Singapore, 2018, pp. 33–37. doi: 10.1007/978-981-10-7419-6_6.

[6] A. H. Sabry, O. I. Dallal Bashi, N. H. Nik Ali, and Y. Mahmood Al Kubaisi, “Lung disease recognition methods using audio-based analysis with machine learning,” Heliyon, vol. 10, no. 4, p. e26218, Feb. 2024, doi: 10.1016/j.heliyon.2024.e26218.

[7] Y. Gong, Y.-A. Chung, and J. Glass, “AST: Audio Spectrogram Transformer,” in Interspeech 2021, ISCA, Aug. 2021, pp. 571–575. doi: 10.21437/Interspeech.2021-698.
