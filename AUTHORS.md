The MuReNN package was created in 2020 by Vincent Lostanlen, a scientist at CNRS.

MuReNN implements a dual-tree complex wavelet transform (DTCWT) in 1-D by
depending on the pytorch_wavelets package, which implements a 2-D DTCWT among
other transforms. The pytorch_wavelets package was created in 2018 by Fergal
Cotter, then a PhD student at the university of Cambridge.

The source code of pytorch_wavelets is available under the MIT license at:
https://github.com/fbcotter/pytorch_wavelets

For more information on the 2-D DTCWT and its application to deep learning,
please read Fergal Cotter's PhD dissertation:
https://www.repository.cam.ac.uk/handle/1810/306661

Note that pytorch_wavelets is itself based on a MATLAB implementation of the
DTCWT by Nick Kingsbury, Cambridge University. Below is a quote from the original
README file, signed by Nick Kingsbury in June 2003:

    Further information on the DT CWT can be obtained from papers downloadable
    from my website (given below). The best tutorial is in the 1999 Royal
    Society Paper. In particular this explains the conversion between 'real'
    quad-number subimages and pairs of complex subimages. The Q-shift filters
    are explained in the ICIP 2000 paper and in more detail in the May 2001
    paper for the Journal on Applied and Computational Harmonic Analysis.

    This code is copyright and is supplied free of charge for research purposes
    only. In return for supplying the code, all I ask is that, if you use the
    algorithms, you give due reference to this work in any papers that you write
    and that you let me know if you find any good applications for the DT CWT.
    If the applications are good, I would be very interested in collaboration.
    I accept no liability arising from use of these algorithms.


We refer to the source code of pytorch_wavelets for more details on the
conditions of use of the DTCWT.
