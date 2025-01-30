========
pydatemm
========

.. image:: singlebat_traj_ccg_output.png
   :width: 600
   
Update 2025 January: *This package is not very user-installable yet, and documentation may be patchy. This is a proof-of-principle implementation that didn't go too far when I realised the DATEMM/CCG class methods are rather slow when you need to start tracking multiple sources ( >=3 sources) with multiple microphones (>=12 mics). The code *does* work though, and the main issue is that there are a lot of false positives (and some false negatives too) - which makes cleaning the data hard unless you have another sensor modality to go along (e.g. camera based trajectories).*



A package to localise sources in overlapping multi-channel audio data. 

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
The CCG algorithm finds the largest subsets of unique compatible fundamental-loops, that thus result in >=4 channel TDOA graphs that can then be used to localise sources. 


Installation & usage
--------------------
This package is rather under-documented from a user-perspective, and active development was stopped ~2023. 
 
The heart of the package is the ```generate_candidate_sources``` function in the ```source_generation``` module. 

For examples scripts that use the ```generate_candidate_sources``` see the ```examples/``` folder, and in specific:
	* ```ushichka_2018-08-17_speakerplayback.py```

For examples of bash scripts (ending with ```.sh```) that were used to run the code on computing clusters with the Slurm job-manager check out:
	* ```1529543496_origxyz.sh``` 






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
