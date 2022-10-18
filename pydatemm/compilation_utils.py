#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C++ Compilation Utilities
=========================
Finds libiomp5 library and runs cppyy based compilation.

@author: thejasvi
"""
import cppyy 
import subprocess
import sys
import os 

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
        raise NotImplementedError(f"{sys.platform} OS not handled currently")

def load_and_compile_cpp_code():
    libiomp5_path = get_libiomp5_path()
    os.environ['EXTRA_CLING_ARGS'] = '-fopenmp -O2'
    not_compiled = True
    for each in libiomp5_path:
        try:
            cppyy.load_library(each)
            cppyy.add_include_path('./eigen/')
            cppyy.include('./sw2002_vectorbased.h')
            cppyy.include('./combineall_cpp/ui_combineall.cpp')
            print(f'Done compiling wiht {each} library')
            not_compiled = False
            break
        except:
            pass
    if not_compiled:
        print('Unable to compile - check if you have libiomp5 or if  the cpp code has already been compiled')
        
