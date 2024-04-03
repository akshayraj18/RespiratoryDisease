---
layout: default
---

# Project Proposal

## Introduction/Background

With the rate of respiratory ailments on the rise, the stress imposed on global health becomes more prevalent [1]. These ailments, if left untreated, can leave an individual with irreversible damage to their respiratory system by impairing lung function. Recent developments in medical technology have yielded datasets of recorded lung sounds, opening up new possibilities for automated lung sound examination. 

We analyze the largest and most prominent of these datasets, ICBHI [4, 5]; this dataset consists of 920 annotated recordings (via 4 different devices) from 126 patients totaling 5.5 hours and 6898 breathing cycles, as well as structured data on each patient’s age, sex, height, weight, and respiratory disease.

## Problem Definition

Motivated by our desire to improve early detection of serious respiratory disease, we analyze audio samples using machine learning methods to differentiate between 6 different kinds of respiratory conditions (COPD, URTI, Bronchiectasis, Pneumonia, Bronchiolitis, and healthy) as a way of pre-screening patients. We believe this project will be best utilized as a web application to expedite triage in medical establishments or to enhance at-home symptom checking. While most published work on the topic has focused on identifying abnormal sounds, such as crackles and wheezes, with high levels of accuracy [1,2,6], we go further and attempt to identify the specific pulmonary disease of a patient given their demographic information and auscultation audio sample.

## Methods and Results (Will be discussed together for each model)

### Preprocessing

Our dataset consists of various audio files in .wav format. Each audio file contains several breathing cycles that may have crackles and/or wheezes. The timestamps for the breathing cycles, as well as whether a crackle or wheeze was detected, are found in the .txt file that corresponds to a given .wav file. The initial phase of our preprocessing involves splitting all the 920 .wav files into separate audio clips given by the timestamps in the corresponding .txt files. We store the 6898 newly preprocessed audio files in a folder called clips_by_cycle. Next, we generate mel spectrograms from these clips. These spectrograms visualize each audio clip in the following manner: the x-axis represents the timestamp of a given sample within the cycle, the y-axis represents a certain frequency bucket, and the intensity of the color of each time-frequency square associated with the spectrograms represents the magnitude of the given frequency at that time. Then, we join the demographic data from demographic_info.txt with our mel spectrogram .jpg files and our .wav clips by cycle on patient ID and cycle number. Additionally, we add a BMI column as we have some child heights and weights. Our final data frame includes patient id, cycle number, start and end timestamps, whether a crackle and/or wheeze was detected in each cycle, chest location and acquisition position, recording equipment, the audio clip for the cycle, the mel spectrogram for the cycle, and BMI, amongst other columns. For future use, we implement an additional step in our preprocessing that can be used for the supervised learning models. We first downsample the files to 4KHz, as the sampling rate is different for each file. Then we use a 5th order butter band-pass filter that removes background noise such as background speech, and heartbeat. We then standardize and normalize the audio clips. These three preprocessing steps are complete on each audio file and add to another folder for future utilization. We then generate spectrograms using these preprocessed audio clips and save these in a folder called normalized_spectrograms. In addition to using these normalized spectrograms and preprocessed audio clips for the supervised learning models, we will compare our findings from the unsupervised GMM model from the old spectrograms and the newly preprocessed ones. 

### GMM

Before we approach the problem of analyzing the audio samples with supervised learning methods, we wish to visualize the distribution of the patients and observe whether there exist any clusters of patients that are predisposed to each of the 8 diseases (7 diseases and healthy) we will be classifying between. We first fit a Gaussian Mixture Model (GMM) on the most relevant demographic features present in the demographic data (namely age and bmi). We choose to use a GMM over other unsupervised learning methods, as GMM is a soft-clustering method; it returns responsibility probabilities that can give us a general idea of how likely a given datapoint (i.e. patient) is to be in each of the 8 clusters (i.e. have each of the 8 conditions), which could prove to be useful for diagnosis.

Since there are 7 diseases and 1 healthy category, we choose to create 8 clusters. This is the resulting plot: 
![GMM Clusters](/images/GMMClusters.png)

![True Diseases](/images/DiseaseClusters.png)
*We also plot the true diseases for each of these patients*

To evaluate the performance of our GMM, we use both an internal measure, silhouette coefficient, and an external measure, completeness score. The silhouette coefficient is intended to measure how well-defined the found clusters are. Our clustering obtained a silhouette coefficient of .376, which is less than the desired threshold of at least 0.5, indicating that the GMM did not return tight clusters that are spread out (i.e. our clusters are not well-defined). The completeness score is intended to measure how well the clusters of our GMM align with the true distribution of the patient diagnoses. Using the diseases as the true labels, we obtain a completeness score of .388. As completeness scores near 1 reflect better clusterings of our data with respect to the true diagnoses, this once again is a poor result. These metrics and the corresponding visualizations suggest that GMM clustering does not provide as much information about more susceptible populations as we would have hoped. However, one interesting observation to note is that patients who have one of LRTI, Bronchiolitis, URTI (or are healthy) tend to lie in a very different area of Age-BMI space than patients with COPD, Asthma, and (oftentimes) pneumonia. Hence, for our next steps, we will experiment with training a GMM on just these two macro-classes of diagnoses to try to extract insights that may be useful in our supervised analysis.

### PCA

These mel spectrograms are 200x500 images; hence, each spectrogram is composed of 100,000 features. In order to greatly reduce the number of features and decrease our future models’ (both the unsupervised and supervised models to be applied on the audio files) complexity, we apply principal component analysis (PCA) on spectrograms. We use PCA on a sample of 1000 spectrograms in order to get an estimate for how many components are needed to retain at least 99% of the original variance. We choose a relatively large cutoff of 99%, as spectrograms contain high-fidelity data that must be kept intact. Below, we visualize the total explained variance ratio as a function of the number of kept principal components (image right-trimmed for brevity).  

![PCA](/images/MelPCA.png)
*Number of components required to explain 99% of the variance: 927 Below, we visualize 5 examples of spectrograms. The top row depicts the 5 original spectrograms (100,000 features), and the bottom row depicts the reconstruction of these spectrograms from the top 927 principal components.*

![PCA Reconstruction](/images/PCAVariance.png)

## Next Steps

Our next steps are to refine the preprocessing for the Butterworth band-pass filter by adjusting the low and high band cut values. We will also implement data augmentation and statistical resampling methods to even out the number of entries per diagnosis. To complete our unsupervised learning analysis, we will also extract MFCCs from the audio files and dimension reduce them using both PCA and t-SNE, then attempt to cluster them by using a GMM. Following this, we will first implement a neural net using convolutional and recurrent layers using PyTorch. Next, we will create an audio spectrogram transformer (AST) using Hugging Face and try an original method (Vision Mamba) on our dataset. Finally, we will build a random forest (RF) to model the mix of categorical and continuous variables in our dataset. We will experiment with ensembles of these models such as feeding the input of the neural net embeddings, scores, and probabilities into the random forest.

## References

[1] S. Bae et al., “Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification,” in INTERSPEECH 2023, ISCA, Aug. 2023, pp. 5436–5440. doi: 10.21437/Interspeech.2023-1426.

[2] S. Gairola, F. Tom, N. Kwatra, and M. Jain, “RespireNet: A Deep Neural Network for Accurately Detecting Abnormal Lung Sounds in Limited Data Setting,” in 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Nov. 2021, pp. 527–530. doi: 10.1109/EMBC46164.2021.9630091.

[3] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-Balanced Loss Based on Effective Number of Samples,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2019, pp. 9260–9269. doi: 10.1109/CVPR.2019.00949.

[4] “Respiratory Sound Database.” Accessed: Feb. 22, 2024. [Online]. Available: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database

[5] B. M. Rocha et al., “Α Respiratory Sound Database for the Development of Automated Classification,” in Precision Medicine Powered by pHealth and Connected Health, vol. 66, N. Maglaveras, I. Chouvarda, and P. De Carvalho, Eds., Singapore: Springer Singapore, 2018, pp. 33–37. doi: 10.1007/978-981-10-7419-6_6.

[6] A. H. Sabry, O. I. Dallal Bashi, N. H. Nik Ali, and Y. Mahmood Al Kubaisi, “Lung disease recognition methods using audio-based analysis with machine learning,” Heliyon, vol. 10, no. 4, p. e26218, Feb. 2024, doi: 10.1016/j.heliyon.2024.e26218.

[7] Y. Gong, Y.-A. Chung, and J. Glass, “AST: Audio Spectrogram Transformer,” in Interspeech 2021, ISCA, Aug. 2021, pp. 571–575. doi: 10.21437/Interspeech.2021-698.
