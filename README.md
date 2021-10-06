# Domain-Agnostic Features for Acoustic SceneClassification using Disentanglement Learning

## Reference

<pre>
 Abeßer, J. & Müller, M. Towards Audio Domain Adaptation for Acoustic Scene Classification using Disentanglement Learning, submitted to: ICASSP 2022
</pre>

## Related Work

* we use pre-computed features & model architecture used in 3 previous papers
    * these are all unsupervised domain adaptation methods
    
<pre>
    Mezza, A. I., Habets, E. A. P., Müller, M., & Sarti, A. (2021).
    #Unsupervised domain adaptation for acoustic scene classification
    using band-wise statistics matching. Proceedings of the European
    Signal Processing Conference (EUSIPCO), 11–15.
    https://doi.org/10.23919/Eusipco47968.2020.9287533"

    Drossos, K., Magron, P., & Virtanen, T. (2019). Unsupervised Adversarial Domain Adaptation based
    on the Wasserstein Distance for Acoustic Scene Classification. Proceedings of the IEEE Workshop
    on Applications of Signal Processing to Audio and Acoustics (WASPAA), 259–263. New Paltz, NY, USA.

    Gharib, S., Drossos, K., Emre, C., Serdyuk, D., & Virtanen, T. (2018). Unsupervised Adversarial Domain
    Adaptation for Acoustic Scene Classification. Proceedings of the Detection and Classification of
    Acoustic Scenes and Events (DCASE). Surrey, UK.

</pre>

## Files

* ```configs.py``` - Training configurations (C0 ... C3M)
* ```generator.py``` - Data generator
* ```losses.py``` - Loss implementations
* ```model.py``` - Function to create dual-input / dual-output model
* ```model_kaggle.py``` - reference CNN model from related work for acoustic scene classification (ASC)
* ```normalization.py``` - Normalization methods (see Mezza et al. above)
* ```params.py``` - General parameters
* ```prediction.py``` - Prediction script to evaluate models on test data
* ```training.py``` - Script to run the model training for 6 different configurations (see Fig. 2 in 
the paper)

## How to run

* create python environment (e.g. with conda), the following versions were used during the paper preparation process
  * librosa==0.8.0
  * matplotlib==3.3.2
  * numpy=1.19.2
  * python=3.7.0
  * scikit-learn==0.23.2
  * tensorflow==2.3.0
  * torch==1.9.0
* set in ```params.py``` the following variables
   * ```dir_feat``` to your local copy of the ```.p``` files from https://zenodo.org/record/1401995 
   * ```dir_target``` to your local output folder
* run ```python training.py && python prediction.py``` on a GPU device to train & evaluate the models

