=====
Usage
=====
Get started using :code:`pydatemm` with a few lines. Importing pydatemm the first time could take some time as the C++ compiler needs to compile all the code. Depending on your system it could take a few seconds to upto a minute. The first run of a C++ based function could also take longer - but the next runs run at 'normal' speed.

.. code-block:: python 

   import pydatemm
   from pydatemmsource_generation import generate_candidate_sources_hybrid

   # load the simulated audio data here, along with the additional parameters
   # needed to get the example running. 
   from pydatemm.sim_data import sim_audio, kwargs

   localisation_outputs = generate_candidate_sources_hybrid(sim_audio, **kwargs)

And there you have it! Since |package| uses C++ to optimise the localisation, the raw outputs
are also C++ objects. :code:`localisation_outputs` is a C++ ``struct``. |package| uses the :code:`cppyy` package
to move between Python and C++. The ``sources`` attribute holds
the a vector of vectors representing each sources x, y, z and time-difference-of-arrival (TDOA) residual in seconds.

.. code-block:: python 

   # Let's see how many sources have been calculated
   print(localisation_outputs.sources.size())
   >>> 1939 
   # Here's an example source output
   print(localisation_outputs.sources[0]) 
   >>> {2.1707984, 12.217783, 1.2226018, 1.3858988e-17}

Here the first three elements correspond to the x, y, and z coordinates. The last element refers
to the TDOA residual (defined in Scheuing & Yang 2008), the lower the better.

Convert to NumPy and off you go
-------------------------------
For most users, the best way to proceed is to convert the :code:`vector<vector<double>>` into a 2D NumPy array.

.. code-block:: python
   
   import numpy as np     
   # convert all the sources into a NumPy array to perform further custom-processing.
   sources =   np.array([each for each in localisation_outputs])
   print(sources[0,:]) 




.. |package| replace:: :code:`pydatemm`
