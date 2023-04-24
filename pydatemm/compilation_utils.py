#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C++ Compilation Utilities
=========================
Finds libiomp5 library and runs cppyy based compilation.

@author: thejasvi
"""
import subprocess
import sys
import os 
import glob
import cppyy_backend.loader as l

def get_libiomp5_path():
    '''
    Tries to find the first found libiomp5 DLL and returns
    the path

    Raises
    ------
    ValueError
        If no libiomp5 DLL can be found
    NotImplementedError
        If non-Linux OS is used.

    Returns
    -------
    str
        Path to the libiomp5 DLL
    '''
    
    if sys.platform == "linux":
        output = subprocess.check_output(["locate", "libiomp5"])
        # parse the output 
        results = str(output)[2:].split("\\n")
        # choose only those with libiomp5.so in them!
        only_libiomp5 = [each for each in results if 'libiomp5.so' in each]
        if len(only_libiomp5)==0:
            raise ValueError('No libiomp5 found')
        # prefer the Anaconda libiomp5 if available - otherwise choose something
        # else
        anaconda_lib = [each for each in only_libiomp5 if 'anaconda' in each]
        if len(anaconda_lib) > 0:
            return anaconda_lib
        else:
            return only_libiomp5
    else:
        #TODO : User needs to have added an environmental variable 
        # for LIBIOMP5PATH 
        # Maybe this is the best default solution actually?
        if sys.platform == 'win32':
            only_libiomp5 = [os.environ['LIBIOMP5PATH']]
            return only_libiomp5
        else:
            raise NotImplementedError(f"{sys.platform} OS not handled currently")

def custom_libiomp5_path():
    '''
    Looks to find the libiomp5.so or .dll through the 
    environmental variable 'LIBIOMP5PATH'
    '''
    try:
        only_libiomp5 = [os.environ['LIBIOMP5PATH']]
        return only_libiomp5
    except:
        raise ValueError('Unable to find environmental variable LIBIOMP5PATH')
        

def get_eigen_path():
    current_module_path = os.path.abspath(__file__)
    current_folder = os.path.split(current_module_path)[0]
    return  os.path.join(current_folder, 'eigen/')

def get_cpp_modules():
    current_module_path = os.path.abspath(__file__)
    current_folder = os.path.split(current_module_path)[0]
    cpp_modules = ['eigen_utils.h',
                   'fast_localiser.cpp',
                   'mpr2003_vectorbased.h',
                   'sw2002_vectorbased.h',
                   'combineall_cpp/ui_combineall.cpp',
                   'graph_manip_ccp.cpp']
    cpp_files = [os.path.join(current_folder, each) for each in cpp_modules]
    return cpp_files



    


def load_and_compile_with_own_flags():
    os.environ['EXTRA_CLING_ARGS'] = '-fopenmp -O2 -g'
    current_folder = os.path.split(os.getcwd())[0]
    pch_folder = os.path.join(current_folder, 'cling_pch/')
    if os.path.exists(pch_folder):
        pass
    else:
        os.mkdir(pch_folder)
    os.environ['CLING_STANDARD_PCH'] = os.path.join(pch_folder, 'std_with_openmp.pch')
    
    import cppyy
    try:
        cppyy.load_library(get_libiomp5_path()[0])
    except:
        raise ValueError(f'Could not load libiomp5 with {get_libiomp5_path()[0]}')
    cppyy.add_include_path(get_eigen_path())
    for each in get_cpp_modules():
        cppyy.include(each)

