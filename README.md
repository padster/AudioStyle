# AudioStyle
UBC CPSC 540 Project: Style transfer for Audio

For details, see the paper in this repository:
  https://github.com/padster/AudioStyle/blob/master/paper/CPSC540_FinalReport.pdf

Code uses the following implementation of the Neural Style algorithm, in Lasagne/Theano:
  https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb

Audio processing logic (spectrogram & mfcc) used from:
  https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html

To test yourself:
  python run.py <flags>

Where the available flags are:
  * --cpu (whether to run theano on the CPU, default is GPU)
  * --spec (whether to transfer the spectrogram, default is MFCC)
  * --rowac (whether to include loss for row autocorrelation)
  * --colac (whether to include loss for column autocorrelation)
