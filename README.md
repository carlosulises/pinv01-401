<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center">NNO</h1>
  <h3 align="center">Neural Network Optimizer</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
      <ul>
        <li><a href="#-ependencies">Dependencies</a></li>
      </ul>
    </li>
    <li>
      <a href="#how-to-use">How to use</a>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About the Project

The framework dynamically generates and trained models of artificial neural networks base in a previous configuration. To perform the work, the tool loads the dataset into a dynamically constructed model and trained in intervals in two ways. In the first way, the model is trained in the traditional way as everyone knows and in the second way training is done by layer and intervals. Once all the layers have been trained a model is generated that would be the optimized model.

<p align="right">(<a href="#top">back to top</a>)</p>


### Dependencies

This project was developed with the following dependencies:

* [TensorFlow](https://www.tensorflow.org)
* [Keras](https://keras.io/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scipy](https://scipy.org/)
* [h5py](https://www.h5py.org/)
* [skopt](https://scikit-optimize.github.io/stable/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## How to use

To use the tool, all the dependencies mentioned above must be installed. After cloning the project from the repository, the config.ini file must be modified, which is the place where the layer types are assigned and the file where all the configurations of the architecture and dataset.

The most used layers and their properties are considered in a convolutional network. In case you want to add more types of layers that are not contemplated in the tool, you can add them by developing small plugins for these layers, as you can see in the layers folder.

To start the execution, you must run the main.py file that executes the dynamic construction of the architecture, training the model and analyzing the data generated in the predictions.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Carlos Valdez - [LinkedIn](https://www.linkedin.com/in/carlos-valdez-85537016b) - carulises95@gmail.com

Project link: [https://gitlab.com/deeplearningviewer/dlviewer](https://gitlab.com/deeplearningviewer/dlviewer)

<p align="right">(<a href="#top">back to top</a>)</p>
