# The Art of Transcription 

## Abstract 

The use of deep learning is constantly growing, new complex applications are being solved thanks to the use of neural networks. For example, audio signals is a type of data that is being used in neural networks to achieve new solutions. The aim of this project is to study to what extent a neuronal network is able to transcript polyphonic musical notes from an audio signal. Using Musicnet dataset as the initial corpus, a sound signal pre-processing technique has been applied to create a new representation from the time domain to the frequency domain using Constant-Q Transforms (CQT), a variant of the well-known Fast Fourier transforms to encode a richer representation of the data for the neural network. Consequently, from the new representation of the signal, images have been produced to constitute the input data of a Convolutional Neuronal Network (CNN). Results show that the model struggles to generalise for the test set due to the complexity of the songs and the abundance of possible musical notes that can be played at the same time. Nevertheless, the model is able to achieve a 23\% accuracy predicting successfully the notes that appear in the song.

## Paper

This paper has been carried out by Manuel Mínguez for Automatic Speech Recognition course at Radboud University.

- ManuelMínguez-TheArtOfTranscription.pdf
