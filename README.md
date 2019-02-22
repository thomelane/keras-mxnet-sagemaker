# Keras-MXNet on SageMaker

An example of training and deploying a Keras-MXNet model on Amazon SageMaker.

As of 22nd Feburary 2019, uses latest SageMaker MXNet containers (v1.3.0), script mode training interface and updated deployment functions.

### Contents

* Use `nb_train.ipynb` for in-notebook training of Keras-MXNet CNN model.
* Use `sm_train_deploy.ipynb` for SageMaker training job and endpoint deployment of above model.
    * `keras_mnist.py` is script used for training and inference
    * `input.html` is for interactive MNIST input inside notebook.
