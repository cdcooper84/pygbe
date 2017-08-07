#!/usr/bin/env python
'''
  Copyright (C) 2017 by Christopher Cooper

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
'''
"""
This script reads Tinker input (xyz and key files)
and spits out a xyzr file, readeable by msms
"""

import sys
import numpy
from numpy import *

file_in = sys.argv[1]
file_xyz = file_in+'.xyz'
file_key = file_in+'.key'
file_out = file_in+'.xyzr'


with open(file_xyz, 'r') as f:
    N = int(f.readline().split()[0])

x = numpy.zeros(N)
y = numpy.zeros(N)
z = numpy.zeros(N)
r = numpy.zeros(N)
atom_type  = numpy.chararray(N, itemsize=10)

i = 0
for line in file(file_xyz):
    line = line.split()

    if len(line)>2:
        x[i] = numpy.float64(line[2])
        y[i] = numpy.float64(line[3])
        z[i] = numpy.float64(line[4])
        atom_type[i] = line[5]
        i+=1

atom_class = {}
vdw_radii = {}

for line in file(file_key):
    line = line.split()

    if len(line)>0:
        if line[0].lower()=='atom':
            atom_class[line[1]] = line[2]

        if line[0].lower()=='vdw':
            vdw_radii[line[1]] = numpy.float64(line[2])
                
for i in range(N):
    r[i] = vdw_radii[atom_class[atom_type[i]]] 
    
data = numpy.zeros((N,4))

data[:,0] = x[:]
data[:,1] = y[:]
data[:,2] = z[:]
data[:,3] = r[:]

numpy.savetxt(file_out, data, fmt='%5.6f')
