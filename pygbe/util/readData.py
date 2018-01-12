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
    alpha = numpy.zeros((N,3,3))
    atom_type  = numpy.chararray(N, itemsize=10)
    connections = numpy.empty(N, dtype=object)
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
    polarizability = {}
    charge = {}
    dipole = {}
    quadrupole = {}
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

            if line[0].lower()=='polarize':
                polarizability[line[1]] = REAL(line[2])

            if line[0].lower()=='multipole' or (multipole_flag>0 and multipole_flag<5):

                if multipole_flag == 0:
                    key = line[1]
                    z_axis = line[2]
                    x_axis = line[3]
                    if len(line)>5:
                        y_axis = line[4]
                    else:
                        y_axis = 'NA'

                    multipole_list.append((key, z_axis, x_axis, y_axis))

                    charge[(key, z_axis, x_axis, y_axis)] = REAL(line[-1])
                if multipole_flag == 1:
                    dipole[(key, z_axis, x_axis, y_axis)] = numpy.array([REAL(line[0]), REAL(line[1]), REAL(line[2])]) 
                if multipole_flag == 2:
                    quadrupole[(key, z_axis, x_axis, y_axis)] = numpy.zeros((3,3))
                    quadrupole[(key, z_axis, x_axis, y_axis)][0,0] = REAL(line[0])
                if multipole_flag == 3:
                    quadrupole[(key, z_axis, x_axis, y_axis)][1,0] = REAL(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis)][0,1] = REAL(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis)][1,1] = REAL(line[1])
                if multipole_flag == 4:
                    quadrupole[(key, z_axis, x_axis, y_axis)][2,0] = REAL(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis)][0,2] = REAL(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis)][2,1] = REAL(line[1])
                    quadrupole[(key, z_axis, x_axis, y_axis)][1,2] = REAL(line[1])
                    quadrupole[(key, z_axis, x_axis, y_axis)][2,2] = REAL(line[2])
                    multipole_flag = -1            

                multipole_flag += 1
                
    for i in range(N):
        alpha[i,:,:] = numpy.identity(3)*polarizability[atom_type[i]]

#       filter possible multipoles by atom type
        atom_possible = []
        for j in range(len(multipole_list)):
            if atom_type[i] == multipole_list[j][0]:
                atom_possible.append(multipole_list[j])

#       filter possible multipoles by z axis defining atom (needs to be bonded)
#       only is atom_possible has more than 1 alternative
        if len(atom_possible)>1:
            zaxis_possible = []
            for j in range(len(atom_possible)):
                for k in connections[i]:
                    neigh_type = atom_type[k]
                    if neigh_type == atom_possible[j][1]:
                        zaxis_possible.append(atom_possible[j])

#           filter possible multipoles by x axis defining atom (no need to be bonded)
#           only if zaxis_possible has more than 1 alternative
            if len(zaxis_possible)>1:
                neigh_type = []
                for j in range(len(zaxis_possible)):
                    neigh_type.append(zaxis_possible[j][2])

                xaxis_possible_atom = []
                for j in range(N):
                    if atom_type[j] in neigh_type:
                        xaxis_possible_atom.append(j)

                dist = numpy.linalg.norm(pos[i,:] - pos[xaxis_possible_atom,:], axis=1)


                xaxis_at_index = numpy.where(numpy.abs(dist - numpy.min(dist))<1e-12)[0][0]
                xaxis_at = xaxis_possible_atom[xaxis_at_index]

#               just check if it's not a connection
                if xaxis_at not in connections[i]:
#                    print 'For atom %i+1, x axis define atom is %i+1, which is not bonded'%(i,xaxis_at)
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
        
        q[i] = charge[xaxis_possible[-1]]
        p[i,:] = dipole[xaxis_possible[-1]]
        Q[i,:,:] = quadrupole[xaxis_possible[-1]]
        
    return pos, q, p, Q, alpha, N

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
