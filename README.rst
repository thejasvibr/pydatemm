========
pydatemm
========
A package to localise sources in overlapping multi-channel audio data. 


`Update (2024 January)
Development on this package has been paused as I realised phase-difference based algorithms may do better
with the kinds of overlapping, reverberant audio that I am dealing with.`

The name `pydatemm` refers to the DATEMM (Scheuing & Yang 2008) algorithm that was the original implementation. 
The current version of this package implements the Compatibility-Conflict-Graph (CCG) (Kreißig & Yang 2013) algorithm.
The CCG and linear algebra localisation scripts are optimised for speed and written in C++ (python-C++ communication
through the :code:`cppyy` package.

Overview
--------
The Compatibility-Conflict-Graph algorithm by Kreißig & Yang is the successor of the DATEMM algorithm (Scheuing & Yang 2008), 
which builds zero-sum time-difference-of-arrival (TDOA) triplets, and 'fuses' compatible triplets together to make a bigger TDOA
graph. 

In CCG, zero-sum triplets are first built systematically by investigating the presence of zero-sum 'fundamental loops'. Then, based
on the values and node identities, all zero-sum loops are compared with each other and a large N_loop x N_loop binary matrix is filled
to indicate compatibility or a conflict. A compatibility indicates two triplets can be fused together, while a conflict indicates they can't. 
The CCG algorithm finds the largest subsets of unique compatible fundamental-loops, that thus result in >=4 channel TDOA graphs that can
then be used to localise sources. 

References
----------
* Scheuing, J., & Yang, B. (2008). Disambiguation of TDOA estimation for multiple sources in reverberant environments. IEEE transactions on audio, speech, and language processing, 16(8), 1479-1489.
* Kreißig, M., & Yang, B. (2013, May). Fast and reliable TDOA assignment in multi-source reverberant environments. In 2013 IEEE International Conference on Acoustics, Speech and Signal Processing (pp. 355-359). IEEE.


License
-------
* Free software: MIT license
* Documentation: https://pydatemm.readthedocs.io.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
