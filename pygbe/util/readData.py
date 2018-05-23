'''
  Copyright (C) 2013 by Christopher Cooper, Lorena Barba

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

import numpy
import os

def readVertex2(filename, REAL):
    x = []
    y = []
    z = []
    for line in file(filename):
        line = line.split()
        x0 = line[0]
        y0 = line[1]
        z0 = line[2]
        x.append(REAL(x0))
        y.append(REAL(y0))
        z.append(REAL(z0))

    x = numpy.array(x)
    y = numpy.array(y)
    z = numpy.array(z)
    vertex = numpy.zeros((len(x),3))
    vertex[:,0] = x
    vertex[:,1] = y
    vertex[:,2] = z
    return vertex 

def readVertex(filename, REAL):
    full_path = os.environ.get('PYGBE_PROBLEM_FOLDER')+'/'
    X = numpy.loadtxt(full_path + filename, dtype=REAL)
    vertex = X[:,0:3]

    return vertex

def readTriangle2(filename):
    triangle = []

    for line in file(filename):
        line = line.split()
        v1 = line[0]
        v2 = line[2] # v2 and v3 are flipped to match my sign convention!
        v3 = line[1]
        triangle.append([int(v1)-1,int(v2)-1,int(v3)-1])
        # -1-> python starts from 0, matlab from 1

    triangle = numpy.array(triangle)

    return triangle

def readTriangle(filename, surf_type):
    full_path = os.environ.get('PYGBE_PROBLEM_FOLDER')+'/'
    X = numpy.loadtxt(full_path + filename, dtype=int)
    triangle = numpy.zeros((len(X),3), dtype=int)
#    if surf_type<=10:
    if surf_type=='internal_cavity':
        triangle[:,0] = X[:,0]
        triangle[:,1] = X[:,1] 
        triangle[:,2] = X[:,2]
    else:
        triangle[:,0] = X[:,0]
        triangle[:,1] = X[:,2] # v2 and v3 are flipped to match my sign convention!
        triangle[:,2] = X[:,1]

    triangle -= 1

    return triangle

def readCheck(aux, REAL):
    # check if it is not reading more than one term
    cut = [0]
    i = 0
    for c in aux[1:]:
        i += 1
        if c=='-':
            cut.append(i)
    cut.append(len(aux))
    X = numpy.zeros(len(cut)-1)
    for i in range(len(cut)-1):
        X[i] = REAL(aux[cut[i]:cut[i+1]])

    return X

def find_multipole(multipole_list, connections, atom_type, pos, i, N):


#   filter possible multipoles by atom type
    atom_possible = []
    for j in range(len(multipole_list)):
        if atom_type[i] == multipole_list[j][0]:
            atom_possible.append(multipole_list[j])

#   filter possible multipoles by z axis defining atom (needs to be bonded)
#   only is atom_possible has more than 1 alternative
    if len(atom_possible)>1:
        zaxis_possible = []
        for j in range(len(atom_possible)):
            for k in connections[i]:
                neigh_type = atom_type[k]
                if neigh_type == atom_possible[j][1]:
                    zaxis_possible.append(atom_possible[j])

#       filter possible multipoles by x axis defining atom (no need to be bonded)
#       only if zaxis_possible has more than 1 alternative
        if len(zaxis_possible)>1:
            neigh_type = []
            for j in range(len(zaxis_possible)):
                neigh_type.append(zaxis_possible[j][2])

            xaxis_possible_atom = []
            for j in range(N):
                if atom_type[j] in neigh_type and i!=j:
                    xaxis_possible_atom.append(j)

            dist = numpy.linalg.norm(pos[i,:] - pos[xaxis_possible_atom,:], axis=1)


            xaxis_at_index = numpy.where(numpy.abs(dist - numpy.min(dist))<1e-12)[0][0]
            xaxis_at = xaxis_possible_atom[xaxis_at_index]

#           just check if it's not a connection
            if xaxis_at not in connections[i]:
#                print 'For atom %i+1, x axis define atom is %i+1, which is not bonded'%(i,xaxis_at)
                for jj in connections[i]:
                    if jj in xaxis_possible_atom:
                        print 'For atom %i+1, there was a bonded connnection available for x axis, but was not used'%(i)

            xaxis_type = atom_type[xaxis_at]

            xaxis_possible = []
            for j in range(len(zaxis_possible)):
                if xaxis_type == zaxis_possible[j][2]:
                    xaxis_possible.append(zaxis_possible[j])

            if len(xaxis_possible)==0:
                print 'For atom %i+1 there is no possible multipole'%i
            if len(xaxis_possible)>1:
                print 'For atom %i+1 there is more than 1 possible multipole, use last one'%i

        else:
            xaxis_possible = zaxis_possible

    else:
        xaxis_possible = atom_possible
    
    multipole = xaxis_possible[-1]

    return multipole

def read_tinker(filename, REAL):
    """
    Reads input file from tinker
    Input:
    -----
    filename: (string) file name without xyz or key extension
    REAL    : (string) precision, double or float

    Returns:
    -------
    pos: Nx3 array with position of multipoles
    q  : array size N with charges (monopoles)
    p  : array size Nx3 with dipoles
    Q  : array size Nx3x3 with quadrupoles
    alpha: array size Nx3x3 with polarizabilities
            (tinker considers an isotropic value, not tensor)
    N  : (int) number of multipoles
    """

    file_xyz = filename+'.xyz'
    file_key = filename+'.key'

    with open(file_xyz, 'r') as f:
        N = int(f.readline().split()[0])

    pos   = numpy.zeros((N,3))
    q     = numpy.zeros(N)
    p     = numpy.zeros((N,3))
    Q     = numpy.zeros((N,3,3))
#    test  = numpy.zeros((N,2))
    alpha = numpy.zeros((N,3,3))
    thole = numpy.zeros(N)
    mass  = numpy.zeros(N)
    atom_type  = numpy.chararray(N, itemsize=10)
    connections = numpy.empty(N, dtype=object)
    polar_group = -numpy.ones(N, dtype=numpy.int32)
    N_connections = 0
    header = 0
    for line in file(file_xyz):
        line = line.split()

        if header==1:
            atom_number = int(line[0])-1
            pos[atom_number,0] = REAL(line[2])
            pos[atom_number,1] = REAL(line[3])
            pos[atom_number,2] = REAL(line[4])
            atom_type[atom_number] = line[5]
            connections[atom_number] = numpy.zeros(len(line)-6, dtype=int)
            N_connections += len(line)-6
            for i in range(6, len(line)):
                connections[atom_number][i-6] = int(line[i]) - 1 

        header = 1

    atom_class = {}
    atom_mass = {}
    polarizability = {}
    thole_factor = {}
    charge = {}
    dipole = {}
    quadrupole = {}
    polar_group_list = {}
    multipole_list = []
    multipole_flag = 0

    with open(file_key, 'r') as f:
        line = f.readline().split()
        if line[0]=='parameters':
            file_key = line[1]
        print ('Reading parameters from '+file_key)

    for line in file(file_key):
        line = line.split()

        if len(line)>0:
            if line[0].lower()=='atom':
                atom_class[line[1]] = line[2]
                atom_mass[line[1]] = REAL(line[-2])

            if line[0].lower()=='polarize':
                polarizability[line[1]] = REAL(line[2])
                thole_factor[line[1]] = REAL(line[3])
                polar_group_list[line[1]] = numpy.chararray(len(line)-4, itemsize=10)
                polar_group_list[line[1]][:] = line[4:]

            if line[0].lower()=='mpole-12-scale':
                m12scale = REAL(line[1])
            if line[0].lower()=='mpole-13-scale':
                m13scale = REAL(line[1])
            if line[0].lower()=='mpole-14-scale':
                m14scale = REAL(line[1])
            if line[0].lower()=='mpole-15-scale':
                m15scale = REAL(line[1])
            if line[0].lower()=='polar-12-scale':
                p12scale = REAL(line[1])
            if line[0].lower()=='polar-13-scale':
                p13scale = REAL(line[1])
            if line[0].lower()=='polar-14-scale':
                p14scale = REAL(line[1])
            if line[0].lower()=='polar-15-scale':
                p15scale = REAL(line[1])

            if line[0].lower()=='multipole' or (multipole_flag>0 and multipole_flag<5):

                if multipole_flag == 0:
                    key = line[1]
                    z_axis = line[2]
                    x_axis = line[3]

                    if len(line)>5:
                        y_axis = line[4]
                    else:
                        y_axis = '0'

                    axis_type = 'z_then_x'
                    if REAL(z_axis)==0:
                        axis_type = 'None'
                    if REAL(z_axis)!=0 and REAL(x_axis)==0: # not implemented yet
                        axis_type = 'z_only'
                    if REAL(z_axis)<0 or REAL(x_axis)<0:
                        axis_type = 'bisector'
                    if REAL(x_axis)<0 and REAL(y_axis)<0: # not implemented yet
                        axis_type = 'z_bisect'
                    if REAL(z_axis)<0 and REAL(x_axis)<0 and REAL(y_axis)<0: # not implemented yet
                        axis_type = '3_fold'
                    
                    # Remove negative defining atom types 
                    if z_axis[0]=='-': 
                        z_axis = z_axis[1:]
                    if x_axis[0]=='-': 
                        x_axis = x_axis[1:]
                    if y_axis[0]=='-': 
                        y_axis = y_axis[1:]

                    multipole_list.append((key, z_axis, x_axis, y_axis, axis_type))

                    charge[(key, z_axis, x_axis, y_axis, axis_type)] = REAL(line[-1])
                if multipole_flag == 1:
                    dipole[(key, z_axis, x_axis, y_axis, axis_type)] = numpy.array([REAL(line[0]), REAL(line[1]), REAL(line[2])]) 
                if multipole_flag == 2:
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)] = numpy.zeros((3,3))
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][0,0] = REAL(line[0])
                if multipole_flag == 3:
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][1,0] = REAL(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][0,1] = REAL(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][1,1] = REAL(line[1])
                if multipole_flag == 4:
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][2,0] = REAL(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][0,2] = REAL(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][2,1] = REAL(line[1])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][1,2] = REAL(line[1])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][2,2] = REAL(line[2])
                    multipole_flag = -1            

                multipole_flag += 1
                
    polar_group_counter = 0
    for i in range(N):
#       Get polarizability
        alpha[i,:,:] = numpy.identity(3)*polarizability[atom_type[i]]

#       Get Thole factor
        thole[i] = thole_factor[atom_type[i]]

#       Get mass
        mass[i] = atom_mass[atom_type[i]]

#       Find atom polarization group
        if polar_group[i]==-1:
#           Check with connections if there is a group member already assigned
            for j in connections[i]:
                if atom_type[j] in polar_group_list[atom_type[i]][:] and polar_group[j]!=-1:
                    if polar_group[i]==-1:
                        polar_group[i] = polar_group[j]
                    elif polar_group[i]!=polar_group[j]:
                        print 'double polarization group assigment here!'
#           if no other group members are found, create a new group
            if polar_group[i]==-1:
                polar_group[i] = numpy.int32(polar_group_counter)
                polar_group_counter += 1

#           Now, assign group number to connections in the same group
            for j in connections[i]:
                if atom_type[j] in polar_group_list[atom_type[i]][:]:
                    if polar_group[j]==-1:
                        polar_group[j] = polar_group[i]
                    elif polar_group[j]!=polar_group[i]: 
                        print 'double polarization group assigment here too!'

        multipole = find_multipole(multipole_list, connections, atom_type, pos, i, N)


#       Find local axis
#       Find z defining atom (needs to be bonded)
        z_atom = -1
        for k in connections[i]:
            neigh_type = atom_type[k]
            if neigh_type == multipole[1]:
                if z_atom == -1:
                    z_atom = k
                else:
                    print 'Two (or more) possible z defining atoms for atom %i+1, using the first one'%i

#       Find x defining atom (no need to be bonded)
#       First, look within 1-2 bonded atoms
        x_atom = -1

        for k in connections[i]:
            neigh_type = atom_type[k]
            if neigh_type == multipole[2] and k!=z_atom:
                if x_atom == -1:
                    x_atom = k
                else:
                    print 'Two (or more) possible x defining atoms in 1-2 for atom %i+1, using the first one'%i

#       Next, look within 1-3 bonded atoms
        if x_atom==-1:
            for k in connections[i]:
                for l in connections[k]:
                    neigh_type = atom_type[l]
                    if neigh_type == multipole[2] and l!=i and l!=z_atom:
                        if x_atom == -1:
                            x_atom = l
                        else:
                            print 'Two (or more) possible x defining atoms in 1-3 for atom %i+1, using the first one'%i

#       Else, look within nonbonded atoms
        if x_atom==-1:
            neigh_type = multipole[2]
            x_possible_atom = []
            for j in range(N):
                if atom_type[j] == neigh_type and i != j and j != z_atom:
                    x_possible_atom.append(j)

            if len(x_possible_atom)>0:
                dist = numpy.linalg.norm(pos[i,:] - pos[x_possible_atom,:], axis=1)

                x_atom_index = numpy.where(numpy.abs(dist - numpy.min(dist))<1e-12)[0][0]
                x_atom = x_possible_atom[x_atom_index]

#       just check if it's not a connection
#        if x_atom not in connections[i]:
#            for jj in connections[i]:
#                if jj in x_possible_atom and jj!=z_atom:
#                    print 'For atom %i+1, there was a bonded atom that could have been x-defining, but is not'

        if z_atom==-1 or x_atom==-1: # for example, in the sphere case
            i_local = numpy.array([1,0,0])
            j_local = numpy.array([0,1,0])
            k_local = numpy.array([0,0,1])

        else:
            r12 = pos[z_atom,:] - pos[i,:]
            r13 = pos[x_atom,:] - pos[i,:]
            if multipole[4]=='z_then_x':
                k_local = r12/numpy.linalg.norm(r12) 
                i_local = (r13 - numpy.dot(r13,k_local)*k_local)/numpy.linalg.norm(r13 - numpy.dot(r13,k_local)*k_local)
                j_local = numpy.cross(k_local, i_local)

            elif multipole[4]=='bisector':
                k_local = r12/numpy.linalg.norm(r12) + r13/numpy.linalg.norm(r13) 
                k_local = k_local/numpy.linalg.norm(k_local)
                i_local = (r13 - numpy.dot(r13,k_local)*k_local)/numpy.linalg.norm(r13 - numpy.dot(r13,k_local)*k_local)
                j_local = numpy.cross(k_local, i_local)
           
#       Assign charge
        q[i] = charge[multipole]
        
#       Find rotation matrix
        A = numpy.identity(3)
        A[:,0] = i_local
        A[:,1] = j_local
        A[:,2] = k_local

        bohr = 0.52917721067
#       Assign dipole
        p[i,:] = numpy.dot(A, dipole[multipole])*bohr
#        print p[i,:]
#        print dipole[multipole]*bohr

#       Assign quadrupole
        for ii in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        Q[i,ii,j] += A[ii,k]*A[j,m]*quadrupole[multipole][k,m]*bohr**2/3. # OJO /3 (See Stone's book)

#   Connections list
#   1-2 connections (already computed, just put into 1D array)
    connections_12 = numpy.zeros(N_connections, dtype=numpy.int32)
    pointer_connections_12 = numpy.zeros(N+1, dtype=numpy.int32)    # pointer to beginning of interaction list
    for i in range(N):
        pointer_connections_12[i+1] = pointer_connections_12[i] + len(connections[i])
        start = pointer_connections_12[i]
        stop = pointer_connections_12[i+1]
        connections_12[start:stop] = connections[i]

#   1-3 connections
    connections_13 = numpy.zeros(N_connections*N_connections/N, dtype=numpy.int32)
    pointer_connections_13 = numpy.zeros(N+1, dtype=numpy.int32)    # pointer to beginning of interaction list
    for i in range(N):
        possible_connections = numpy.concatenate(connections[connections[i]])
        possible_connections = numpy.unique(possible_connections) # filter out repeated connections 
        index_self = numpy.where(possible_connections==i)[0] # remove self atom
        possible_connections = numpy.delete(possible_connections, index_self)
        pointer_connections_13[i+1] = pointer_connections_13[i] + len(possible_connections) 
    
        start = pointer_connections_13[i]
        end   = pointer_connections_13[i+1]
        connections_13[start:end] = possible_connections 

    connections_13 = connections_13[:pointer_connections_13[-1]]


    ''' No need!
#   Find group connections (1-2) 
    N_group = numpy.max(polar_group)+1
    connections_group = numpy.empty(N_group, dtype=object)
    N_connections_group = numpy.zeros(N_group, dtype=numpy.int32)
    for i in range(N_group):
        connections_group[i] = numpy.array([], dtype=numpy.int32)

    for i in range(N):
        for j in connections[i]:
            if polar_group[i]!=polar_group[j] and (polar_group[j] not in connections_group[polar_group[i]]):
                connections_group[polar_group[i]] = numpy.append(connections_group[polar_group[i]], polar_group[j])
                N_connections_group[polar_group[i]] += 1 

#   Put connections in 1D array
    pointer_connections_12 = numpy.zeros(N_group+1, dtype=numpy.int32) # Pointer to beginning of interaciton list
    connections_group_12 = numpy.zeros(numpy.sum(N_connections_group), dtype=numpy.int32)
    for i in range(N_group):
        pointer_connections_12[i+1] = pointer_connections_12[i]+N_connections_group[i]
        connections_group_12[pointer_connections_12[i]:pointer_connections_12[i+1]] = connections_group[i]

#   Find group 1-3, 1-4 and 1-5 connections (check if can be done with numpy.unique)
    pointer_connections_13 = numpy.zeros(N_group+1, dtype=numpy.int32) # Pointer to beginning of interaciton list
    pointer_connections_14 = numpy.zeros(N_group+1, dtype=numpy.int32) # Pointer to beginning of interaciton list
    pointer_connections_15 = numpy.zeros(N_group+1, dtype=numpy.int32) # Pointer to beginning of interaciton list

#   Oversize arrays to avoid append
    connections_group_13 = numpy.zeros(N_group*N_group, dtype=numpy.int32)
    connections_group_14 = numpy.zeros(N_group*N_group, dtype=numpy.int32)
    connections_group_15 = numpy.zeros(N_group*N_group, dtype=numpy.int32)
    for i in range(N_group):

#       1-3 group connections (connection of 1-2 connections)
        possible_connections = numpy.concatenate(connections_group[connections_group[i]])
        possible_connections = numpy.unique(possible_connections) # filter out repeated groups
        index_self = numpy.where(possible_connections==i)[0] # remove self group
        possible_connections = numpy.delete(possible_connections, index_self)
        pointer_connections_13[i+1] = pointer_connections_13[i] + len(possible_connections) 
        
        start = pointer_connections_13[i]
        end   = pointer_connections_13[i+1]
        connections_group_13[start:end] = possible_connections 
        
#       1-4 group connections (connections of 1-3 connections)
        possible_connections = numpy.concatenate(connections_group[connections_group_13[start:end]])
        possible_connections = numpy.unique(possible_connections) # filter out repeated groups
        start_12 = pointer_connections_12[i]
        end_12   = pointer_connections_12[i+1]
        index_12 = numpy.where(possible_connections==connections_group_12[start_12:end_12])[0] # remove groups already in 1-2
        possible_connections = numpy.delete(possible_connections, index_12)
        pointer_connections_14[i+1] = pointer_connections_14[i] + len(possible_connections) 
        
        start = pointer_connections_14[i]
        end   = pointer_connections_14[i+1]
        connections_group_14[start:end] = possible_connections 
 
#       1-5 group connections (connections of 1-4 connections)
        possible_connections = numpy.concatenate(connections_group[connections_group_14[start:end]])
        possible_connections = numpy.unique(possible_connections) # filter out repeated groups
        start_13 = pointer_connections_13[i]
        end_13   = pointer_connections_13[i+1]
        index_13 = numpy.where(possible_connections==connections_group_13[start_13:end_13])[0] # remove groups already in 1-3
        possible_connections = numpy.delete(possible_connections, index_13)
        pointer_connections_15[i+1] = pointer_connections_15[i] + len(possible_connections) 
        
        start = pointer_connections_15[i]
        end   = pointer_connections_15[i+1]
        connections_group_15[start:end] = possible_connections 

    connections_group_13 = connections_group_13[:pointer_connections_13[-1]]
    connections_group_14 = connections_group_14[:pointer_connections_14[-1]]
    connections_group_15 = connections_group_15[:pointer_connections_15[-1]]
    '''

    return pos, q, p, Q, alpha, mass, polar_group, thole, \
           connections_12, connections_13, \
           pointer_connections_12, pointer_connections_13, \
           p12scale, p13scale, N 


def read_tinker_pqr(filename, REAL):
    """
    Reads input file from tinker
    Input:
    -----
    filename: (string) file name without xyz or key extension
    REAL    : (string) precision, double or float

    Returns:
    -------
    pos: Nx3 array with position of multipoles
    q  : array size N with charges (monopoles)
    p  : array size Nx3 with dipoles
    Q  : array size Nx3x3 with quadrupoles
    alpha: array size Nx3x3 with polarizabilities
            (tinker considers an isotropic value, not tensor)
    N  : (int) number of multipoles
    """

    file_xyz = filename+'.xyz'
    file_key = filename+'.key'
    file_pqr = filename+'.pqr.tinker' # self generated file with multipoles in global frame

    with open(file_xyz, 'r') as f:
        N = int(f.readline().split()[0])

    pos   = numpy.zeros((N,3))
    q     = numpy.zeros(N)
    p     = numpy.zeros((N,3))
    Q     = numpy.zeros((N,3,3))
    alpha = numpy.zeros((N,3,3))
    thole = numpy.zeros(N)
    mass  = numpy.zeros(N)
    radii = numpy.zeros(N)
    atom_type  = numpy.chararray(N, itemsize=10)
    connections = numpy.empty(N, dtype=object)
    polar_group = -numpy.ones(N, dtype=int)
    header = 0
    for line in file(file_xyz):
        line = line.split()

        if header==1:
            atom_number = int(line[0])-1
            pos[atom_number,0] = REAL(line[2])
            pos[atom_number,1] = REAL(line[3])
            pos[atom_number,2] = REAL(line[4])
            atom_type[atom_number] = line[5]
            connections[atom_number] = numpy.zeros(len(line)-6, dtype=int)
            for i in range(6, len(line)):
                connections[atom_number][i-6] = int(line[i]) - 1 

        header = 1

    atom_class = {}
    atom_mass = {}
    polarizability = {}
    thole_factor = {}
    polar_group_list = {}
    multipole_list = []
    vdw_radii = {}
    multipole_flag = 0

    with open(file_key, 'r') as f:
        line = f.readline().split()
        if line[0]=='parameters':
            file_key = line[1]
        print ('Reading parameters from '+file_key)

    for line in file(file_key):
        line = line.split()

        if len(line)>0:
            if line[0].lower()=='atom':
                atom_class[line[1]] = line[2]
                atom_mass[line[1]] = REAL(line[-2])

            if line[0].lower()=='polarize':
                polarizability[line[1]] = REAL(line[2])
                thole_factor[line[1]] = REAL(line[3])
                polar_group_list[line[1]] = numpy.chararray(len(line)-4, itemsize=10)
                polar_group_list[line[1]][:] = line[4:]
                
            if line[0].lower()=='vdw':
                vdw_radii[line[1]] = numpy.float64(line[2])

    for line_full in file(file_pqr):
        line_aux = line_full.split()
        if len(line_aux)<17:
            line = []
            for i in range(len(line_aux)):
                if "-" in line_aux[i][1:]:
                    word = line_aux[i][1:].split("-")
                    line.append(line_aux[i][0]+word[0])
                    for j in range(len(word)-1):
                        line.append("-"+word[j+1])
                else:
                    line.append(line_aux[i])
        else:
            line = line_aux
            

        if len(line)>0:
            atom_number = int(line[0])-1
            q[atom_number] = REAL(line[4])
            p[atom_number,0] = REAL(line[5])
            p[atom_number,1] = REAL(line[6])
            p[atom_number,2] = REAL(line[7])
            Q[atom_number,0,0] = REAL(line[8])
            Q[atom_number,0,1] = REAL(line[9])
            Q[atom_number,0,2] = REAL(line[10])
            Q[atom_number,1,0] = REAL(line[11])
            Q[atom_number,1,1] = REAL(line[12])
            Q[atom_number,1,2] = REAL(line[13])
            Q[atom_number,2,0] = REAL(line[14])
            Q[atom_number,2,1] = REAL(line[15])
            Q[atom_number,2,2] = REAL(line[16])
            
                
    polar_group_counter = 0
    for i in range(N):
#       Get polarizability
        alpha[i,:,:] = numpy.identity(3)*polarizability[atom_type[i]]

#       Get Thole factor
        thole[i] = thole_factor[atom_type[i]]

#       Get mass
        mass[i] = atom_mass[atom_type[i]]

#       Get VdW radii
        radii[i] = vdw_radii[atom_class[atom_type[i]]]

#       Find atom polarization group
        if polar_group[i]==-1:
#           Check with connections if there is a group member already assigned
            for j in connections[i]:
                if atom_type[j] in polar_group_list[atom_type[i]][:] and polar_group[j]!=-1:
                    if polar_group[i]==-1:
                        polar_group[i] = polar_group[j]
                    elif polar_group[i]!=polar_group[j]:
                        print 'double polarization group assigment here!'
#           if no other group members are found, create a new group
            if polar_group[i]==-1:
                polar_group[i] = polar_group_counter
                polar_group_counter += 1

#           Now, assign group number to connections in the same group
            for j in connections[i]:
                if atom_type[j] in polar_group_list[atom_type[i]][:]:
                    if polar_group[j]==-1:
                        polar_group[j] = polar_group[i]
                    elif polar_group[j]!=polar_group[i]: 
                        print 'double polarization group assigment here too!'

#   Generate pqr file (debugging)
#    f = open(filename+"_test.pqr","w")
#    for i in range(N):
#        f.write('ATOM\t%i\tAAA\tAAA\t1\t%1.5f\t%1.5f\t%1.5f\t%1.5f\t%1.5f\n'%(i+1,pos[i,0],pos[i,1],pos[i,2],q[i],radii[i]))
#    f.close()
#    quit()



    return pos, q, p, Q, alpha, mass, polar_group, thole, N

def readpqr(filename, REAL):

    pos = []
    q   = []
    p   = []
    Q   = []
    alpha = []
    for line in file(filename):
        line = numpy.array(line.split())
        line_aux = []

        if line[0]=='ATOM':
            if len(line)<12:
                n_data = len(line)-6
            else:
                n_data = len(line)-27
            for l in range(n_data):
                aux = line[5+len(line_aux)]
                if len(aux)>14:
                    X = readCheck(aux,REAL)
                    for i in range(len(X)):
                        line_aux.append(X[i])
#                        line_test.append(str(X[i]))
                else:
#                    line_test.append(line[5+len(line_aux)])
                    line_aux.append(REAL(line[5+len(line_aux)]))

            if len(line)>12:
                line_aux.extend(REAL(line[-21:]))

#            line_test.append(line[len(line)-1])
            x = line_aux[0]
            y = line_aux[1]
            z = line_aux[2]
            pos.append([x,y,z])

            if len(line)>12: # if multipole
                q.append(line_aux[3])
                p.append(numpy.array([line_aux[4],line_aux[5],line_aux[6]]))
                Q.append(numpy.reshape(numpy.array([line_aux[7],line_aux[8],line_aux[9],line_aux[10],line_aux[11],line_aux[12],line_aux[13],line_aux[14],line_aux[15]]),(3,3)))
                if len(line_aux)>16: # if polarizable
                    alpha.append(numpy.reshape(numpy.array([line_aux[16],line_aux[17],line_aux[18],line_aux[19],line_aux[20],line_aux[21],line_aux[22],line_aux[23],line_aux[24]]),(3,3)))
            else:
                q.append(line_aux[3])

#           for i in range(10):
#                f.write("%s\t"%line_test[i])
#            f.write("\n")

#    f.close()
#    quit()
    pos = numpy.array(pos)
    q   = numpy.array(q)
    p   = numpy.array(p)
    Q   = numpy.array(Q)
    alpha = numpy.array(alpha)
    Nq  = len(q)

    return pos, q, p, Q, alpha, Nq


def readcrd(filename, REAL):

    pos = []
    q   = []

    start = 0
    for line in file(filename):
        line = numpy.array(line.split())
   
        if len(line)>8 and line[0]!='*':# and start==2:
            x = line[4]
            y = line[5]
            z = line[6]
            q.append(REAL(line[9]))
            pos.append([REAL(x),REAL(y),REAL(z)])
    
        '''
        if len(line)==1:
            start += 1
            if start==2:
                Nq = int(line[0])
        '''
    pos = numpy.array(pos)
    q   = numpy.array(q)
    Nq  = len(q)
    return pos, q, Nq

def readParameters(param, filename):

    val  = []
    for line in file(filename):
        line = line.split()
        val.append(line[1])

    dataType = val[0]      # Data type
    if dataType=='double':
        param.REAL = numpy.float64
    elif dataType=='float':
        param.REAL = numpy.float32

    REAL = param.REAL
    param.K         = int (val[1])      # Gauss points per element
    param.Nk        = int (val[2])      # Number of Gauss points per side 
                                        # for semi analytical integral
    param.K_fine    = int (val[3])      # Number of Gauss points per element 
                                        # for near singular integrals 
    param.threshold = REAL(val[4])      # L/d threshold to use analytical integrals
                                        # Over: analytical, under: quadrature
    param.BSZ       = int (val[5])      # CUDA block size
    param.restart   = int (val[6])      # Restart for GMRES
    param.tol       = REAL(val[7])      # Tolerance for GMRES
    param.max_iter  = int (val[8])      # Max number of iteration for GMRES
    param.P         = int (val[9])      # Order of Taylor expansion for treecode
    param.eps       = REAL(val[10])     # Epsilon machine
    param.NCRIT     = int (val[11])     # Max number of targets per twig box of tree
    param.theta     = REAL(val[12])     # MAC criterion for treecode
    param.GPU       = int (val[13])     # =1: use GPU, =0 no GPU

    return dataType


def readFields(filename):

    LorY    = []
    pot     = []
    E       = []
    kappa   = []
    charges = []
    coulomb = []
    qfile   = []
    Nparent = []
    parent  = []
    Nchild  = []
    child   = []

    for line in file(filename):
        line = line.split()
        if len(line)>0:
            if line[0]=='FIELD':
                LorY.append(line[1])
                pot.append(line[2])
                E.append(line[3])
                kappa.append(line[4])
                charges.append(line[5])
                coulomb.append(line[6])
                qfile.append(
                    line[7] if line[7] == 'NA'
                    else os.environ.get('PYGBE_PROBLEM_FOLDER')+'/'+line[7]
                )
                Nparent.append(line[8])
                parent.append(line[9])
                Nchild.append(line[10])
                for i in range(int(Nchild[-1])):
                    child.append(line[11+i])

    return LorY, pot, E, kappa, charges, coulomb, qfile, Nparent, parent, Nchild, child

def readSurf(filename):

    files = []
    surf_type = []
    phi0_file = []
    for line in file(filename):
        line = line.split()
        if len(line)>0:
            if line[0]=='FILE':
                files.append(line[1])
                surf_type.append(line[2])
                if line[2]=='dirichlet_surface' or line[2]=='neumann_surface' or line[2]=='neumann_surface_hyper':
                    phi0_file.append(line[3])
                else:
                    phi0_file.append('no_file')

    return files, surf_type, phi0_file
