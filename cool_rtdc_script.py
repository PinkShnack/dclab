# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:01:30 2021

@author: eoconne
"""

import h5py
import numpy as np

# example how to make a h5py file
f = h5py.File("mytestfile.hdf5", "w")

# h5py.File acts like a dict
list(f.keys())

# how to create a dataset from the file
ds = f.create_dataset("mydataset", (100,), dtype="i")
list(f.keys())

ds # this is a HDF5 dataset, similar to numpy shaping

ds.shape
ds.dtype

ds[-1]
ds[...] = np.arange(100)
ds[-1]

# array-style slicing, here we get every ten values
ds[0:100:10]
# same thing using data
ds[ds[0]:ds[-1]:10]
# same thing using shape
ds[0:ds.shape[0]:10]

# dataset attributes
print(ds.shape, ds.size, ds.ndim, ds.dtype) # no nbytes?

# specifying the shape and data when creating dataset
ds = f.create_dataset("mydataset2", (10,10), dtype="i",
                      data=np.arange(100))
ds.dtype
ds[-1]
# have a look at the made up dataset
# easy to do here because it's just a 2D array
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(ds)
plt.show()

# different dtype
ds = f.create_dataset("mydataset3", (10,10), dtype="uint32",
                      data=np.arange(100))
ds.dtype

# slicing is a efficient way of accessing data
ds = f.create_dataset("MyDatset4", (10,10,10), 'f')
ds.shape
ds[0,0,0]
example = ds[0,2:10,1:9:3]
arr = np.arange(10)
arr[1:9:3]

example = ds[:,::2,5]
example = ds[:,::2,1:5] # naturally gives 3 ndim arr
example = ds[0]
example = ds[1,5]
example = ds[1,...]
example = ds[...,6]
example = ds[()]
# these give the same (returns everything)
example = ds[...]
example = ds[:,:,:]
example = ds[:] # not certain about this one

# you can broadcastfor simple slicing
ds[0,:,:] = np.arange(10)  # Broadcasts to (10,10)
ds[:,:,:] = np.zeros((10,10,10))

# for multiple indexing this doesn't work:
ds[0][1] = 3.0  # No effect!
print(ds[0][1])
# because it assigns the 3.0 to the first index which is
# loaded into memory (ds[0])

# Try this instead:
ds[0, 1] = 3.0
print(ds[0, 1]) 
print(ds[0][1])    

# use Dataset.len() instead of len(Dataset) for large datasets

# chunking: use chunk_iter to get an iterator




