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
from math import pi
from tree.FMMutils import computeIndices, precomputeTerms
from tree.direct import coulomb_direct, coulomb_energy_multipole, compute_induced_dipole
from projection import project, project_Kt, get_phir, get_phir_gpu, get_dphirdr, get_dphirdr_gpu, get_d2phirdr2, get_d2phirdr2_gpu
from classes import parameters, index_constant
import time
from util.semi_analytical import GQ_1D
from argparse import ArgumentParser

# PyCUDA libraries
try:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
except:
    pass

## Note: 
##  Remember ordering of equations:
##      Same order as defined in config file,
##      with internal equation first and the external equation.

def selfInterior(surf, s, LorY, param, ind0, timing, kernel):
#    print 'SELF INTERIOR, surface: %i'%s
    K_diag = 2*pi
    V_diag = 0
    IorE   = 1
    K_lyr, V_lyr = project(surf.XinK, surf.XinV, LorY, surf, surf,
                            K_diag, V_diag, IorE, s, param, ind0, timing, kernel)
    v = K_lyr - V_lyr
    return v

def selfExterior(surf, s, LorY, param, ind0, timing, kernel):
#    print 'SELF EXTERIOR, surface: %i, E_hat: %f'%(s, surf.E_hat)
    K_diag = -2*pi
    V_diag = 0.
    IorE   = 2
    K_lyr, V_lyr = project(surf.XinK, surf.XinV, LorY, surf, surf, 
                            K_diag, V_diag, IorE, s, param, ind0, timing, kernel)
    v = -K_lyr + surf.E_hat*V_lyr
    return v, K_lyr, V_lyr

def nonselfExterior(surf, src, tar, LorY, param, ind0, timing, kernel):
#    print 'NONSELF EXTERIOR, source: %i, target: %i, E_hat: %f'%(src,tar, surf[src].E_hat)
    K_diag = 0
    V_diag = 0
    IorE   = 1
    K_lyr, V_lyr = project(surf[src].XinK, surf[src].XinV, LorY, surf[src], surf[tar], 
                            K_diag, V_diag, IorE, src, param, ind0, timing, kernel)
    v = -K_lyr + surf[src].E_hat*V_lyr
    return v

def nonselfInterior(surf, src, tar, LorY, param, ind0, timing, kernel):
#    print 'NONSELF INTERIOR, source: %i, target: %i'%(src,tar)
    K_diag = 0
    V_diag = 0
    IorE   = 2
    K_lyr, V_lyr = project(surf[src].XinK, surf[src].XinV, LorY, surf[src], surf[tar], 
                            K_diag, V_diag, IorE, src, param, ind0, timing, kernel)
    v = K_lyr - V_lyr
    return v

def selfASC(surf, src, tar, LorY, param, ind0, timing, kernel):

    Kt_diag = -2*pi * (surf.Eout+surf.Ein)/(surf.Eout-surf.Ein)
    V_diag = 0
    
    Kt_lyr = project_Kt(surf.XinK, LorY, surf, surf, 
                            Kt_diag, src, param, ind0, timing, kernel)
    
    v = -Kt_lyr
    return v

def gmres_dot (X, surf_array, field_array, ind0, param, timing, kernel):
    
    Nfield = len(field_array)
    Nsurf = len(surf_array)

#   Place weights on corresponding surfaces and allocate memory
    Naux = 0
    for i in range(Nsurf):
        N = len(surf_array[i].triangle)
        if surf_array[i].surf_type=='dirichlet_surface':
            surf_array[i].XinK = numpy.zeros(N) 
            surf_array[i].XinV = X[Naux:Naux+N] 
            Naux += N
        elif surf_array[i].surf_type=='neumann_surface' or surf_array[i].surf_type=='asc_surface':
            surf_array[i].XinK = X[Naux:Naux+N] 
            surf_array[i].XinV = numpy.zeros(N)
            Naux += N
        else:
            surf_array[i].XinK     = X[Naux:Naux+N]
            surf_array[i].XinV     = X[Naux+N:Naux+2*N]
            Naux += 2*N 

        surf_array[i].Xout_int = numpy.zeros(N) 
        surf_array[i].Xout_ext = numpy.zeros(N)

#   Loop over fields
    for F in range(Nfield):

        parent_type = 'no_parent'
        if len(field_array[F].parent)>0:
            parent_type = surf_array[field_array[F].parent[0]].surf_type

        if parent_type=='asc_surface':
#           ASC only for self-interaction so far 
            LorY = field_array[F].LorY
            p = field_array[F].parent[0]
            v = selfASC(surf_array[p], p, p, LorY, param, ind0, timing, kernel)
            surf_array[p].Xout_int += v

        if parent_type!='dirichlet_surface' and parent_type!='neumann_surface' and parent_type!='asc_surface':
            LorY = field_array[F].LorY
            param.kappa = field_array[F].kappa
#           print '\n---------------------'
#           print 'REGION %i, LorY: %i, kappa: %f'%(F,LorY,param.kappa)

#           if parent surface -> self interior operator
            if len(field_array[F].parent)>0:
                p = field_array[F].parent[0]
                v = selfInterior(surf_array[p], p, LorY, param, ind0, timing, kernel)
                surf_array[p].Xout_int += v
                
#           if child surface -> self exterior operator + sibling interaction
#           sibling interaction: non-self exterior saved on exterior vector
            if len(field_array[F].child)>0:
                C = field_array[F].child
                for c1 in C:
                    v,t1,t2 = selfExterior(surf_array[c1], c1, LorY, param, ind0, timing, kernel)
                    surf_array[c1].Xout_ext += v
                    for c2 in C:
                        if c1!=c2:
                            v = nonselfExterior(surf_array, c2, c1, LorY, param, ind0, timing, kernel)
                            surf_array[c1].Xout_ext += v

#           if child and parent surface -> parent-child and child-parent interaction
#           parent->child: non-self interior saved on exterior vector 
#           child->parent: non-self exterior saved on interior vector
            if len(field_array[F].child)>0 and len(field_array[F].parent)>0:
                p = field_array[F].parent[0]
                C = field_array[F].child
                for c in C:
                    v = nonselfExterior(surf_array, c, p, LorY, param, ind0, timing, kernel)
                    surf_array[p].Xout_int += v
                    v = nonselfInterior(surf_array, p, c, LorY, param, ind0, timing, kernel)
                    surf_array[c].Xout_ext += v
     
#   Gather results into the result vector
    MV = numpy.zeros(len(X))
    Naux = 0
    for i in range(Nsurf):
        N = len(surf_array[i].triangle)
        if surf_array[i].surf_type=='dirichlet_surface':
            MV[Naux:Naux+N]     = surf_array[i].Xout_ext*surf_array[i].Precond[0,:] 
            Naux += N
        elif surf_array[i].surf_type=='neumann_surface':
            MV[Naux:Naux+N]     = surf_array[i].Xout_ext*surf_array[i].Precond[0,:] 
            Naux += N
        elif surf_array[i].surf_type=='asc_surface':
            MV[Naux:Naux+N]     = surf_array[i].Xout_int*surf_array[i].Precond[0,:] 
            Naux += N
        else:
            MV[Naux:Naux+N]     = surf_array[i].Xout_int*surf_array[i].Precond[0,:] + surf_array[i].Xout_ext*surf_array[i].Precond[1,:] 
            MV[Naux+N:Naux+2*N] = surf_array[i].Xout_int*surf_array[i].Precond[2,:] + surf_array[i].Xout_ext*surf_array[i].Precond[3,:] 
            Naux += 2*N

    return MV

def generateRHS(field_array, surf_array, param, kernel, timing, ind0):
    F = numpy.zeros(param.Neq)

#   Point charge contribution to RHS
    for j in range(len(field_array)):
        Nq = len(field_array[j].q)
        if Nq>0:
#           First look at CHILD surfaces
            for s in field_array[j].child:          # Loop over surfaces
#           Locate position of surface s in RHS
                s_start = 0
                for ss in range(s):
                    if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[ss].surf_type=='asc_surface':
                        s_start += len(surf_array[ss].xi)
                    else:
                        s_start += 2*len(surf_array[ss].xi)

                s_size = len(surf_array[s].xi)

                aux = numpy.zeros(len(surf_array[s].xi))
                for i in range(Nq):
                    dx_pq = surf_array[s].xi - field_array[j].xq[i,0] 
                    dy_pq = surf_array[s].yi - field_array[j].xq[i,1]
                    dz_pq = surf_array[s].zi - field_array[j].xq[i,2]
                    R_pq = numpy.sqrt(dx_pq*dx_pq + dy_pq*dy_pq + dz_pq*dz_pq)

                    if surf_array[s].surf_type=='asc_surface':
                        aux -= field_array[j].q[i]/(R_pq*R_pq*R_pq) * (dx_pq*surf_array[s].normal[:,0] \
                                                                    + dy_pq*surf_array[s].normal[:,1] \
                                                                    + dz_pq*surf_array[s].normal[:,2])
                    else:
                        aux += field_array[j].q[i]/(field_array[j].E*R_pq) # Point charge

                        if par_reac.args.polarizable: # if polarizable multipoles
                            if len(field_array[j].p)>0:                        # Dipole component
                                p_tot = field_array[j].p[i] + field_array[j].p_pol[i]
                                aux1 = numpy.array([dx_pq, dy_pq, dz_pq])/(R_pq*R_pq*R_pq) 
                                aux += numpy.dot(p_tot, aux1)/(field_array[j].E)

                            if len(field_array[j].Q)>0:                        # Quadrupole component
                                aux1 = numpy.array([[dx_pq*dx_pq, dx_pq*dy_pq, dx_pq*dz_pq],\
                                                    [dy_pq*dx_pq, dy_pq*dy_pq, dy_pq*dz_pq],\
                                                    [dz_pq*dx_pq, dz_pq*dy_pq, dz_pq*dz_pq]])/(2*R_pq**5)
                                aux += 6*numpy.tensordot(field_array[j].Q[i], aux1)/(field_array[j].E) # OJO x6!!!

#               For CHILD surfaces, q contributes to RHS in 
#               EXTERIOR equation (hence Precond[1,:] and [3,:])
    
#               No preconditioner
#                F[s_start:s_start+s_size] += aux

#               With preconditioner
#               If surface is dirichlet or neumann it has only one equation, affected by Precond[0,:]
#               We do this only here (and not in the parent case) because interaction of charges 
#               with dirichlet or neumann surface happens only for the surface as a child surfaces.
                if surf_array[s].surf_type=='dirichlet_surface' or surf_array[s].surf_type=='neumann_surface' or  surf_array[s].surf_type=='asc_surface':
                    F[s_start:s_start+s_size] += aux*surf_array[s].Precond[0,:]
                else:
                    F[s_start:s_start+s_size] += aux*surf_array[s].Precond[1,:]
                    F[s_start+s_size:s_start+2*s_size] += aux*surf_array[s].Precond[3,:]

#           Now look at PARENT surface
            if len(field_array[j].parent)>0:
#           Locate position of surface s in RHS
                s = field_array[j].parent[0]
                s_start = 0
                for ss in range(s):
                    if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[ss].surf_type=='asc_surface':
                        s_start += len(surf_array[ss].xi)
                    else:
                        s_start += 2*len(surf_array[ss].xi)

                s_size = len(surf_array[s].xi)

                aux = numpy.zeros(len(surf_array[s].xi))
                for i in range(Nq):
                    dx_pq = surf_array[s].xi - field_array[j].xq[i,0] 
                    dy_pq = surf_array[s].yi - field_array[j].xq[i,1]
                    dz_pq = surf_array[s].zi - field_array[j].xq[i,2]
                    R_pq = numpy.sqrt(dx_pq*dx_pq + dy_pq*dy_pq + dz_pq*dz_pq)

                    if surf_array[s].surf_type=='asc_surface':
                        aux -= field_array[j].q[i]/(R_pq*R_pq*R_pq) * (dx_pq*surf_array[s].normal[:,0] \
                                                                    + dy_pq*surf_array[s].normal[:,1] \
                                                                    + dz_pq*surf_array[s].normal[:,2])
                    else:
                        aux += field_array[j].q[i]/(field_array[j].E*R_pq) # Point charge


                        if param.args.polarizable: # if polarizable multipoles
                            if len(field_array[j].p)>0:                        # Dipole component
                                p_tot = field_array[j].p[i] + field_array[j].p_pol[i]
                                aux1 = numpy.array([dx_pq, dy_pq, dz_pq])/(R_pq*R_pq*R_pq) 
                                aux += numpy.dot(p_tot, aux1)/(field_array[j].E)

                            if len(field_array[j].Q)>0:                        # Quadrupole component
                                aux1 = numpy.array([[dx_pq*dx_pq, dx_pq*dy_pq, dx_pq*dz_pq],\
                                                    [dy_pq*dx_pq, dy_pq*dy_pq, dy_pq*dz_pq],\
                                                    [dz_pq*dx_pq, dz_pq*dy_pq, dz_pq*dz_pq]])/(2*R_pq**5)
                                aux += 6*numpy.tensordot(field_array[j].Q[i], aux1)/(field_array[j].E)  # OJO x6!!

#               No preconditioner
#                F[s_start:s_start+s_size] += aux
#               With preconditioner
                if surf_array[s].surf_type=='asc_surface':
                    F[s_start:s_start+s_size] += aux*surf_array[s].Precond[0,:]
                else:
                    F[s_start:s_start+s_size] += aux*surf_array[s].Precond[0,:]
                    F[s_start+s_size:s_start+2*s_size] += aux*surf_array[s].Precond[2,:]

#   Dirichlet/Neumann contribution to RHS
    for j in range(len(field_array)):

        dirichlet = []
        neumann   = []
        LorY = field_array[j].LorY

#       Find Dirichlet and Neumann surfaces in region
#       Dirichlet/Neumann surfaces can only be child of region,
#       no point on looking at parent surface
        for s in field_array[j].child:
            if surf_array[s].surf_type=='dirichlet_surface':
                dirichlet.append(s)
            elif surf_array[s].surf_type=='neumann_surface':
                neumann.append(s)
    
        if len(neumann)>0 or len(dirichlet)>0:
            
#           First look at influence on SIBLING surfaces
            for s in field_array[j].child:

                param.kappa = field_array[j].kappa

#               Effect of dirichlet surfaces
                for sd in dirichlet:
                    K_diag = -2*pi*(sd==s)
                    V_diag = 0
                    IorE   = 2
                    K_lyr, V_lyr = project(surf_array[sd].phi0, numpy.zeros(len(surf_array[sd].xi)), LorY, surf_array[sd], 
                            surf_array[s], K_diag, V_diag, IorE, sd, param, ind0, timing, kernel)

#                   Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2*len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)

#                   if s is a charged surface, the surface has only one equation, 
#                   else, s has 2 equations and K_lyr affects the external
#                   equation (SIBLING surfaces), which is placed after the internal 
#                   one, hence Precond[1,:] and Precond[3,:].
                    if surf_array[s].surf_type=='dirichlet_surface' or surf_array[s].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                        F[s_start:s_start+s_size] += K_lyr * surf_array[s].Precond[0,:]
                    else:
                        F[s_start:s_start+s_size] += K_lyr * surf_array[s].Precond[1,:]
                        F[s_start+s_size:s_start+2*s_size] += K_lyr * surf_array[s].Precond[3,:]

#               Effect of neumann surfaces
                for sn in neumann:
                    K_diag = 0
                    V_diag = 0
                    IorE   = 2
                    K_lyr, V_lyr = project(numpy.zeros(len(surf_array[sn].phi0)), surf_array[sn].phi0, LorY, surf_array[sn], 
                            surf_array[s], K_diag, V_diag, IorE, sn, param, ind0, timing, kernel)

#                   Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2*len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)

#                   if s is a charge surface, the surface has only one equation, 
#                   else, s has 2 equations and V_lyr affects the external
#                   equation, which is placed after the internal one, hence
#                   Precond[1,:] and Precond[3,:].
                    if surf_array[s].surf_type=='dirichlet_surface' or surf_array[s].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                        F[s_start:s_start+s_size] += -V_lyr * surf_array[s].Precond[0,:]
                    else:
                        F[s_start:s_start+s_size] += -V_lyr * surf_array[s].Precond[1,:]
                        F[s_start+s_size:s_start+2*s_size] += -V_lyr * surf_array[s].Precond[3,:]

#           Now look at influence on PARENT surface
#           The dirichlet/neumann surface will never be the parent, 
#           since we are not solving for anything inside them.
#           Then, in this case we will not look at self interaction,
#           which is dealt with by the sibling surface section
            if len(field_array[j].parent)==1:

                s = field_array[j].parent[0]


#               Effect of dirichlet surfaces
                for sd in dirichlet:
                    K_diag = 0  
                    V_diag = 0
                    IorE   = 2
                    K_lyr, V_lyr = project(surf_array[sd].phi0, numpy.zeros(len(surf_array[sd].xi)), LorY, surf_array[sd], 
                            surf_array[s], K_diag, V_diag, IorE, sd, param, ind0, timing, kernel)

#                   Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2*len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)

#                   Surface s has 2 equations and K_lyr affects the internal
#                   equation, hence Precond[0,:] and Precond[2,:].
                    F[s_start:s_start+s_size] += K_lyr * surf_array[s].Precond[0,:]
                    F[s_start+s_size:s_start+2*s_size] += K_lyr * surf_array[s].Precond[2,:]

#               Effect of neumann surfaces
                for sn in neumann:
                    K_diag = 0
                    V_diag = 0
                    IorE   = 2
                    K_lyr, V_lyr = project(numpy.zeros(len(surf_array[sn].phi0)), surf_array[sn].phi0, LorY, surf_array[sn], 
                            surf_array[s], K_diag, V_diag, IorE, sn, param, ind0, timing, kernel)

#                   Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2*len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)

#                   Surface s has 2 equations and K_lyr affects the internal
#                   equation, hence Precond[0,:] and Precond[2,:].
                    F[s_start:s_start+s_size] += -V_lyr * surf_array[s].Precond[0,:]
                    F[s_start+s_size:s_start+2*s_size] += -V_lyr * surf_array[s].Precond[2,:]

    return F

def generateRHS_gpu(field_array, surf_array, param, kernel, timing, ind0):

    F = numpy.zeros(param.Neq)
    REAL = param.REAL
    computeRHS_gpu = kernel.get_function("compute_RHS")
    computeRHSKt_gpu = kernel.get_function("compute_RHSKt")
    computeRHS_gpu_dipole = kernel.get_function("compute_RHS_dipole")
    computeRHS_gpu_quadrupole = kernel.get_function("compute_RHS_quadrupole")
    for j in range(len(field_array)):
        Nq = len(field_array[j].q)
        if Nq>0:

            if param.args.polarizable:
                if len(field_array[j].p)>0:  # Update dipole component on GPU (done every self-iteration)
                    p_tot = field_array[j].p + field_array[j].p_pol
                    px_tot_gpu = gpuarray.zeros(len(field_array[j].xq), dtype=REAL)
                    py_tot_gpu = gpuarray.zeros(len(field_array[j].xq), dtype=REAL)
                    pz_tot_gpu = gpuarray.zeros(len(field_array[j].xq), dtype=REAL)
                    px_tot_gpu = gpuarray.to_gpu(p_tot[:,0].astype(REAL))
                    py_tot_gpu = gpuarray.to_gpu(p_tot[:,1].astype(REAL))
                    pz_tot_gpu = gpuarray.to_gpu(p_tot[:,2].astype(REAL))
                    field_array[j].px_gpu = gpuarray.to_gpu(field_array[j].p[:,0].astype(REAL))
                    field_array[j].py_gpu = gpuarray.to_gpu(field_array[j].p[:,1].astype(REAL))
                    field_array[j].pz_gpu = gpuarray.to_gpu(field_array[j].p[:,2].astype(REAL))
        
#           First for CHILD surfaces
            for s in field_array[j].child[:]:       # Loop over surfaces
#           Locate position of surface s in RHS
                s_start = 0
                for ss in range(s):
                    if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[ss].surf_type=='asc_surface':
                        s_start += len(surf_array[ss].xi)
                    else:
                        s_start += 2*len(surf_array[ss].xi)

                s_size = len(surf_array[s].xi)
                Nround = len(surf_array[s].twig)*param.NCRIT

                GSZ = int(numpy.ceil(float(Nround)/param.NCRIT)) # CUDA grid size

                if surf_array[s].surf_type!='asc_surface':
                    F_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                    computeRHS_gpu(F_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, field_array[j].q_gpu,
                                surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, numpy.int32(Nq), 
                                REAL(field_array[j].E), numpy.int32(param.NCRIT), numpy.int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 

                    aux = numpy.zeros(Nround)
                    F_gpu.get(aux)

                    if param.args.polarizable:
                        if len(field_array[j].p)>0:                        # Dipole component
                            F_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                            
                            computeRHS_gpu_dipole(F_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, 
                                        px_tot_gpu, py_tot_gpu, pz_tot_gpu,
                                        surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, numpy.int32(Nq), 
                                        REAL(field_array[j].E), numpy.int32(param.NCRIT), numpy.int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 

                            aux2 = numpy.zeros(Nround)
                            F_gpu.get(aux2)
                            aux += aux2

                        if len(field_array[j].Q)>0:                        # quadrupole component
                            F_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                            
                            computeRHS_gpu_quadrupole(F_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, 
                                        field_array[j].Qxx_gpu, field_array[j].Qxy_gpu, field_array[j].Qxz_gpu,
                                        field_array[j].Qyx_gpu, field_array[j].Qyy_gpu, field_array[j].Qyz_gpu,
                                        field_array[j].Qzx_gpu, field_array[j].Qzy_gpu, field_array[j].Qzz_gpu,
                                        surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, numpy.int32(Nq), 
                                        REAL(field_array[j].E), numpy.int32(param.NCRIT), numpy.int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 

                            aux2 = numpy.zeros(Nround)
                            F_gpu.get(aux2)
                            aux += aux2

                else: 
                    Fx_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                    Fy_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                    Fz_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                    computeRHSKt_gpu(Fx_gpu, Fy_gpu, Fz_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, field_array[j].q_gpu,
                                surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, numpy.int32(Nq), 
                                REAL(field_array[j].E), numpy.int32(param.NCRIT), numpy.int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 
                    aux_x = numpy.zeros(Nround)
                    aux_y = numpy.zeros(Nround)
                    aux_z = numpy.zeros(Nround)
                    Fx_gpu.get(aux_x)
                    Fy_gpu.get(aux_y)
                    Fz_gpu.get(aux_z)

                    aux = aux_x[surf_array[s].unsort]*surf_array[s].normal[:,0] + \
                          aux_y[surf_array[s].unsort]*surf_array[s].normal[:,1] + \
                          aux_z[surf_array[s].unsort]*surf_array[s].normal[:,2]

#               For CHILD surfaces, q contributes to RHS in 
#               EXTERIOR equation (hence Precond[1,:] and [3,:])
    
#               No preconditioner
#                F[s_start:s_start+s_size] += aux
#               With preconditioner
#                F[s_start:s_start+s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[1,:]
#               F[s_start+s_size:s_start+2*s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[3,:]

#               With preconditioner
#               If surface is dirichlet or neumann it has only one equation, affected by Precond[0,:]
#               We do this only here (and not in the parent case) because interaction of charges 
#               with dirichlet or neumann surface happens only for the surface as a child surfaces.
                if surf_array[s].surf_type=='dirichlet_surface' or surf_array[s].surf_type=='neumann_surface':
                    F[s_start:s_start+s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[0,:]
                elif surf_array[s].surf_type=='asc_surface':
                    F[s_start:s_start+s_size] += aux*surf_array[s].Precond[0,:]
                else:
                    F[s_start:s_start+s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[1,:]
                    F[s_start+s_size:s_start+2*s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[3,:]


#           Now for PARENT surface
            if len(field_array[j].parent)>0:
                s = field_array[j].parent[0]

#           Locate position of surface s in RHS
                s_start = 0
                for ss in range(s):
                    if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[ss].surf_type=='asc_surface':
                        s_start += len(surf_array[ss].xi)
                    else:
                        s_start += 2*len(surf_array[ss].xi)

                s_size = len(surf_array[s].xi)
                Nround = len(surf_array[s].twig)*param.NCRIT

                GSZ = int(numpy.ceil(float(Nround)/param.NCRIT)) # CUDA grid size
                
                if surf_array[s].surf_type!='asc_surface':
                    F_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                    computeRHS_gpu(F_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, field_array[j].q_gpu,
                                surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, numpy.int32(Nq), 
                                REAL(field_array[j].E), numpy.int32(param.NCRIT), numpy.int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 

                    aux = numpy.zeros(Nround)
                    F_gpu.get(aux)

                    if len(field_array[j].p)>0:                        # Dipole component
                        F_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                        
                        computeRHS_gpu_dipole(F_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, 
                                    px_tot_gpu, py_tot_gpu, pz_tot_gpu,
                                    surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, numpy.int32(Nq), 
                                    REAL(field_array[j].E), numpy.int32(param.NCRIT), numpy.int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 

                        aux2 = numpy.zeros(Nround)
                        F_gpu.get(aux2)
                        aux += aux2

                    if len(field_array[j].Q)>0:                        # quadrupole component
                        F_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                        
                        computeRHS_gpu_quadrupole(F_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, 
                                    field_array[j].Qxx_gpu, field_array[j].Qxy_gpu, field_array[j].Qxz_gpu,
                                    field_array[j].Qyx_gpu, field_array[j].Qyy_gpu, field_array[j].Qyz_gpu,
                                    field_array[j].Qzx_gpu, field_array[j].Qzy_gpu, field_array[j].Qzz_gpu,
                                    surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, numpy.int32(Nq), 
                                    REAL(field_array[j].E), numpy.int32(param.NCRIT), numpy.int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 

                        aux2 = numpy.zeros(Nround)
                        F_gpu.get(aux2)
                        aux += aux2

                else:
                    Fx_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                    Fy_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                    Fz_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                    computeRHSKt_gpu(Fx_gpu, Fy_gpu, Fz_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, field_array[j].q_gpu,
                                surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, numpy.int32(Nq), 
                                REAL(field_array[j].E), numpy.int32(param.NCRIT), numpy.int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 
                    aux_x = numpy.zeros(Nround)
                    aux_y = numpy.zeros(Nround)
                    aux_z = numpy.zeros(Nround)
                    Fx_gpu.get(aux_x)
                    Fy_gpu.get(aux_y)
                    Fz_gpu.get(aux_z)

                    aux = aux_x[surf_array[s].unsort]*surf_array[s].normal[:,0] + \
                          aux_y[surf_array[s].unsort]*surf_array[s].normal[:,1] + \
                          aux_z[surf_array[s].unsort]*surf_array[s].normal[:,2]

#               For PARENT surface, q contributes to RHS in 
#               INTERIOR equation (hence Precond[0,:] and [2,:])
    
#               No preconditioner
#                F[s_start:s_start+s_size] += aux
#               With preconditioner
                if surf_array[s].surf_type=='asc_surface':
                    F[s_start:s_start+s_size] += aux*surf_array[s].Precond[0,:]
                else:
                    F[s_start:s_start+s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[0,:]
                    F[s_start+s_size:s_start+2*s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[2,:]

#   Dirichlet/Neumann contribution to RHS
    for j in range(len(field_array)):

        dirichlet = []
        neumann   = []
        LorY = field_array[j].LorY

#       Find Dirichlet and Neumann surfaces in region
#       Dirichlet/Neumann surfaces can only be child of region,
#       no point on looking at parent surface
        for s in field_array[j].child:
            if surf_array[s].surf_type=='dirichlet_surface':
                dirichlet.append(s)
            elif surf_array[s].surf_type=='neumann_surface':
                neumann.append(s)

        if len(neumann)>0 or len(dirichlet)>0:
            
#           First look at influence on SIBLING surfaces
            for s in field_array[j].child:

                param.kappa = field_array[j].kappa

#               Effect of dirichlet surfaces
                for sd in dirichlet:
                    K_diag = -2*pi*(sd==s)
                    V_diag = 0
                    IorE   = 2 
                    K_lyr, V_lyr = project(surf_array[sd].phi0, numpy.zeros(len(surf_array[sd].xi)), LorY, surf_array[sd], 
                            surf_array[s], K_diag, V_diag, IorE, sd, param, ind0, timing, kernel)

#                   Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2*len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)

#                   if s is a charged surface, the surface has only one equation, 
#                   else, s has 2 equations and K_lyr affects the external
#                   equation, which is placed after the internal one, hence
#                   Precond[1,:] and Precond[3,:].
                    if surf_array[s].surf_type=='dirichlet_surface' or surf_array[s].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                        F[s_start:s_start+s_size] += K_lyr * surf_array[s].Precond[0,:]
                    else:
                        F[s_start:s_start+s_size] += K_lyr * surf_array[s].Precond[1,:]
                        F[s_start+s_size:s_start+2*s_size] += K_lyr * surf_array[s].Precond[3,:]

#               Effect of neumann surfaces
                for sn in neumann:
                    K_diag = 0
                    V_diag = 0
                    IorE = 2
                    K_lyr, V_lyr = project(numpy.zeros(len(surf_array[sn].phi0)), surf_array[sn].phi0, LorY, surf_array[sn], 
                            surf_array[s], K_diag, V_diag, IorE, sn, param, ind0, timing, kernel)

#                   Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2*len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)

#                   if s is a charged surface, the surface has only one equation, 
#                   else, s has 2 equations and V_lyr affects the external
#                   equation, which is placed after the internal one, hence
#                   Precond[1,:] and Precond[3,:].
                    if surf_array[s].surf_type=='dirichlet_surface' or surf_array[s].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                        F[s_start:s_start+s_size] += -V_lyr * surf_array[s].Precond[0,:]
                    else:
                        F[s_start:s_start+s_size] += -V_lyr * surf_array[s].Precond[1,:]
                        F[s_start+s_size:s_start+2*s_size] += -V_lyr * surf_array[s].Precond[3,:]

#           Now look at influence on PARENT surface
#           The dirichlet/neumann surface will never be the parent, 
#           since we are not solving for anything inside them.
#           Then, in this case we will not look at self interaction.
            if len(field_array[j].parent)==1:

                s = field_array[j].parent[0]


#               Effect of dirichlet surfaces
                for sd in dirichlet:
                    K_diag = 0  
                    V_diag = 0
                    IorE   = 1
                    K_lyr, V_lyr = project(surf_array[sd].phi0, numpy.zeros(len(surf_array[sd].xi)), LorY, surf_array[sd], 
                            surf_array[s], K_diag, V_diag, IorE, sd, param, ind0, timing, kernel)

#                   Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2*len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)

#                   Surface s has 2 equations and K_lyr affects the internal
#                   equation, hence Precond[0,:] and Precond[2,:].
                    F[s_start:s_start+s_size] += K_lyr * surf_array[s].Precond[0,:]
                    F[s_start+s_size:s_start+2*s_size] += K_lyr * surf_array[s].Precond[2,:]

#               Effect of neumann surfaces
                for sn in neumann:
                    K_diag = 0
                    V_diag = 0
                    IorE   = 1
                    K_lyr, V_lyr = project(numpy.zeros(len(surf_array[sn].phi0)), surf_array[sn].phi0, LorY, surf_array[sn], 
                            surf_array[s], K_diag, V_diag, IorE, sn, param, ind0, timing, kernel)

#                   Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or surf_array[s].surf_type=='asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2*len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)

#                   Surface s has 2 equations and K_lyr affects the internal
#                   equation, hence Precond[0,:] and Precond[2,:].
                    F[s_start:s_start+s_size] += -V_lyr * surf_array[s].Precond[0,:]
                    F[s_start+s_size:s_start+2*s_size] += -V_lyr * surf_array[s].Precond[2,:]


    return F

def calculateEsolv(surf_array, field_array, param, kernel):

    REAL = param.REAL

    par_reac = parameters()
    par_reac = param
    par_reac.threshold = 0.05
    par_reac.P = 7
    par_reac.theta = 0.0
    par_reac.Nm= (par_reac.P+1)*(par_reac.P+2)*(par_reac.P+3)/6

    ind_reac = index_constant()
    computeIndices(par_reac.P, ind_reac)
    precomputeTerms(par_reac.P, ind_reac)

    par_reac.Nk = 13         # Number of Gauss points per side for semi-analytical integrals

    cal2J = 4.184
    C0 = param.qe**2*param.Na*1e-3*1e10/(cal2J*param.E_0)
    E_solv = []

    ff = -1
    for f in param.E_field:
        parent_type = surf_array[field_array[f].parent[0]].surf_type
        if parent_type != 'dirichlet_surface' and parent_type != 'neumann_surface':

            E_solv_aux = 0
            ff += 1
            print 'Calculating solvation energy for region %i, stored in E_solv[%i]'%(f,ff)
            
            AI_int = 0
            Naux = 0
            phi_reac = numpy.zeros(len(field_array[f].q))

            if par_reac.args.polarizable: # if polarizable multipoles
                dphix_reac = numpy.zeros(len(field_array[f].p))
                dphiy_reac = numpy.zeros(len(field_array[f].p))
                dphiz_reac = numpy.zeros(len(field_array[f].p))
                dphixx_reac = numpy.zeros(len(field_array[f].Q))
                dphixy_reac = numpy.zeros(len(field_array[f].Q))
                dphixz_reac = numpy.zeros(len(field_array[f].Q))
                dphiyx_reac = numpy.zeros(len(field_array[f].Q))
                dphiyy_reac = numpy.zeros(len(field_array[f].Q))
                dphiyz_reac = numpy.zeros(len(field_array[f].Q))
                dphizx_reac = numpy.zeros(len(field_array[f].Q))
                dphizy_reac = numpy.zeros(len(field_array[f].Q))
                dphizz_reac = numpy.zeros(len(field_array[f].Q))

#           First look at CHILD surfaces
#           Need to account for normals pointing outwards
#           and E_hat coefficient (as region is outside and 
#           dphi_dn is defined inside)
            for i in field_array[f].child:
                s = surf_array[i]
                s.xk,s.wk = GQ_1D(par_reac.Nk)
                s.xk = REAL(s.xk)
                s.wk = REAL(s.wk)
                for C in range(len(s.tree)):
                    s.tree[C].M  = numpy.zeros(par_reac.Nm)
                    s.tree[C].Md = numpy.zeros(par_reac.Nm)

                Naux += len(s.triangle)

#               Coefficient to account for dphi_dn defined in
#               interior but calculation done in exterior
                C1 = s.E_hat

                if param.GPU==0:
                    phi_aux, AI = get_phir(s.phi, C1*s.dphi, s, field_array[f].xq, s.tree, par_reac, ind_reac)

                    if par_reac.args.polarizable: # if polarizable multipoles
                        if len(field_array[f].p)>0:
                            dphix_aux, dphiy_aux, dphiz_aux, AI = get_dphirdr (s.phi, C1*s.dphi, s, field_array[f].xq, s.tree, par_reac, ind_reac)
                        if len(field_array[f].Q)>0:
                            dphixx_aux, dphixy_aux, dphixz_aux, \
                            dphiyx_aux, dphiyy_aux, dphiyz_aux, \
                            dphizx_aux, dphizy_aux, dphizz_aux, AI = get_d2phirdr2 (s.phi, C1*s.dphi, s, field_array[f].xq, s.tree, par_reac, ind_reac)
                        
                elif param.GPU==1:
                    phi_aux, AI = get_phir_gpu(s.phi, C1*s.dphi, s, field_array[f], par_reac, kernel)

                    if par_reac.args.polarizable: # if polarizable multipoles
                        if len(field_array[f].p)>0:
                            dphix_aux, dphiy_aux, dphiz_aux, AI = get_dphirdr_gpu (s.phi, C1*s.dphi, s, field_array[f], par_reac, kernel)
                        if len(field_array[f].Q)>0:
                            dphixx_aux, dphixy_aux, dphixz_aux, \
                            dphiyx_aux, dphiyy_aux, dphiyz_aux, \
                            dphizx_aux, dphizy_aux, dphizz_aux, AI = get_d2phirdr2_gpu (s.phi, C1*s.dphi, s, field_array[f], par_reac, kernel)
                        
                

                AI_int += AI
                phi_reac -= phi_aux # Minus sign to account for normal pointing out

                if par_reac.args.polarizable: # if polarizable multipoles
                    if len(field_array[f].p)>0:
                        dphix_reac -= dphix_aux
                        dphiy_reac -= dphiy_aux
                        dphiz_reac -= dphiz_aux

                    if len(field_array[f].Q)>0:
                        dphixx_reac -= dphixx_aux
                        dphixy_reac -= dphixy_aux
                        dphixz_reac -= dphixz_aux
                        dphiyx_reac -= dphiyx_aux
                        dphiyy_reac -= dphiyy_aux
                        dphiyz_reac -= dphiyz_aux
                        dphizx_reac -= dphizx_aux
                        dphizy_reac -= dphizy_aux
                        dphizz_reac -= dphizz_aux

#           Now look at PARENT surface
            if len(field_array[f].parent)>0:
                i = field_array[f].parent[0]
                s = surf_array[i]
                s.xk,s.wk = GQ_1D(par_reac.Nk)
                s.xk = REAL(s.xk)
                s.wk = REAL(s.wk)
                for C in range(len(s.tree)):
                    s.tree[C].M  = numpy.zeros(par_reac.Nm)
                    s.tree[C].Md = numpy.zeros(par_reac.Nm)

                Naux += len(s.triangle)

                if param.GPU==0:
                    phi_aux, AI = get_phir(s.phi, s.dphi, s, field_array[f].xq, s.tree, par_reac, ind_reac)

                    if par_reac.args.polarizable: # if polarizable multipoles
                        if len(field_array[f].p)>0:
                            dphix_aux, dphiy_aux, dphiz_aux, AI = get_dphirdr (s.phi, s.dphi, s, field_array[f].xq, s.tree, par_reac, ind_reac)
                        if len(field_array[f].Q)>0:
                            dphixx_aux, dphixy_aux, dphixz_aux, \
                            dphiyx_aux, dphiyy_aux, dphiyz_aux, \
                            dphizx_aux, dphizy_aux, dphizz_aux, AI = get_d2phirdr2 (s.phi, s.dphi, s, field_array[f].xq, s.tree, par_reac, ind_reac)
                        
                elif param.GPU==1:
                    phi_aux, AI = get_phir_gpu(s.phi, s.dphi, s, field_array[f], par_reac, kernel)

                    if par_reac.args.polarizable: # if polarizable multipoles
                        if len(field_array[f].p)>0:
                            dphix_aux, dphiy_aux, dphiz_aux, AI = get_dphirdr_gpu (s.phi, s.dphi, s, field_array[f], par_reac, kernel)
                        if len(field_array[f].Q)>0:
                            dphixx_aux, dphixy_aux, dphixz_aux, \
                            dphiyx_aux, dphiyy_aux, dphiyz_aux, \
                            dphizx_aux, dphizy_aux, dphizz_aux, AI = get_d2phirdr2_gpu (s.phi, s.dphi, s, field_array[f], par_reac, kernel)
                 
                AI_int += AI
                phi_reac += phi_aux 

                if par_reac.args.polarizable: # if polarizable multipoles
                    if len(field_array[f].p)>0:
                        dphix_reac += dphix_aux
                        dphiy_reac += dphiy_aux
                        dphiz_reac += dphiz_aux
                    if len(field_array[f].Q)>0:
                        dphixx_reac += dphixx_aux
                        dphixy_reac += dphixy_aux
                        dphixz_reac += dphixz_aux
                        dphiyx_reac += dphiyx_aux
                        dphiyy_reac += dphiyy_aux
                        dphiyz_reac += dphiyz_aux
                        dphizx_reac += dphizx_aux
                        dphizy_reac += dphizy_aux
                        dphizz_reac += dphizz_aux

            E_solv_aux += 0.5*C0*numpy.sum(field_array[f].q*phi_reac)
    
            if par_reac.args.polarizable: # if polarizable multipoles
                if len(field_array[f].p)>0:
                    E_solv_aux += 0.5*C0*numpy.sum(field_array[f].p[:,0]*dphix_reac +\
                                                   field_array[f].p[:,1]*dphiy_reac +\
                                                   field_array[f].p[:,2]*dphiz_reac)

                if len(field_array[f].Q)>0:
                    E_solv_aux += 0.5*C0*(1/1.)*numpy.sum( field_array[f].Q[:,0,0]*dphixx_reac +\
                                                           field_array[f].Q[:,0,1]*dphixy_reac +\
                                                           field_array[f].Q[:,0,2]*dphixz_reac +\
                                                           field_array[f].Q[:,1,0]*dphiyx_reac +\
                                                           field_array[f].Q[:,1,1]*dphiyy_reac +\
                                                           field_array[f].Q[:,1,2]*dphiyz_reac +\
                                                           field_array[f].Q[:,2,0]*dphizx_reac +\
                                                           field_array[f].Q[:,2,1]*dphizy_reac +\
                                                           field_array[f].Q[:,2,2]*dphizz_reac ) 
                                                           # OJO: not 1/6

            E_solv.append(E_solv_aux)

            print '%i of %i analytical integrals for phi_reac calculation in region %i'%(AI_int/len(field_array[f].xq),Naux, f)

    return E_solv      

def dissolved_polarizable_dipole(surf_array, field_array, par_reac, ind_reac, kernel):
    """
    Computes the polarizable (induced) dipole in dissolved state. The
    dipole is induced by the coulomb and solvent reaction fields on the
    multipole site.
    Inputs:
    ------
    surf_array : array of classes with surfaces.
    field_array: array of classes with dielectric regions.
    par_reac   : class, parameters relevant to the simulation.
                 These are specific to reaction field calculation
                 as it requires finer parameters.
    ind_reac   : class, contains precomputed indices useful for 
                 treecode calculation
    kernel     : pycuda source module.

    Returns:
    -------
           None: modifies the member p_pol of each class in the field_array
                 array that contains a region with multipoles.
    """
    REAL = par_reac.REAL

    for f in field_array:
        if len(f.q)>0:

            px_pol = numpy.zeros(len(f.xq))
            py_pol = numpy.zeros(len(f.xq))
            pz_pol = numpy.zeros(len(f.xq))

            parent_type = surf_array[f.parent[0]].surf_type
            if parent_type != 'dirichlet_surface' and parent_type != 'neumann_surface':

                AI_int = 0
                Naux = 0

                dphix_reac = numpy.zeros(len(f.p))
                dphiy_reac = numpy.zeros(len(f.p))
                dphiz_reac = numpy.zeros(len(f.p))

#               First look at CHILD surfaces
#               Need to account for normals pointing outwards
#               and E_hat coefficient (as region is outside and 
#               dphi_dn is defined inside)
                for i in f.child:
                    s = surf_array[i]
                    s.xk,s.wk = GQ_1D(par_reac.Nk)
                    s.xk = REAL(s.xk)
                    s.wk = REAL(s.wk)

                    Naux += len(s.triangle)

#                   Coefficient to account for dphi_dn defined in
#                   interior but calculation done in exterior
                    C1 = s.E_hat

                    if par_reac.GPU==0:
                        dphix_aux, dphiy_aux, dphiz_aux, AI = get_dphirdr (s.phi, C1*s.dphi, s, f.xq, s.tree, par_reac, ind_reac)
                    else:
                        dphix_aux, dphiy_aux, dphiz_aux, AI = get_dphirdr_gpu (s.phi, C1*s.dphi, s, f, par_reac, kernel)
                            
                    AI_int += AI

                    dphix_reac -= dphix_aux
                    dphiy_reac -= dphiy_aux
                    dphiz_reac -= dphiz_aux

#               Now look at PARENT surface
                if len(f.parent)>0:
                    i = f.parent[0]
                    s = surf_array[i]
                    s.xk,s.wk = GQ_1D(par_reac.Nk)
                    s.xk = REAL(s.xk)
                    s.wk = REAL(s.wk)

                    Naux += len(s.triangle)

                    if par_reac.GPU==0:
                        dphix_aux, dphiy_aux, dphiz_aux, AI = get_dphirdr (s.phi, s.dphi, s, f.xq, s.tree, par_reac, ind_reac)
                    else:
                        dphix_aux, dphiy_aux, dphiz_aux, AI = get_dphirdr_gpu (s.phi, s.dphi, s, f, par_reac, kernel)
                    
                    AI_int += AI

                    dphix_reac += dphix_aux
                    dphiy_reac += dphiy_aux
                    dphiz_reac += dphiz_aux

                # Computes the coulomb electric field and the induced dipole
                # (induced by both the coulomb and reaction fields)
                if par_reac.GPU==0 or par_reac.GPU==1:
                    px_pol[:] = f.p_pol[:,0]
                    py_pol[:] = f.p_pol[:,1]
                    pz_pol[:] = f.p_pol[:,2]
                    compute_induced_dipole(f.xq[:,0], f.xq[:,1], f.xq[:,2], f.q, 
                                           f.p[:,0], f.p[:,1], f.p[:,2],
                                           px_pol, py_pol, pz_pol,
                                           f.Q[:,0,0], f.Q[:,0,1], f.Q[:,0,2],
                                           f.Q[:,1,0], f.Q[:,1,1], f.Q[:,1,2],
                                           f.Q[:,2,0], f.Q[:,2,1], f.Q[:,2,2],
                                           f.alpha[:,0,0], f.alpha[:,0,1], f.alpha[:,0,2],
                                           f.alpha[:,1,0], f.alpha[:,1,1], f.alpha[:,1,2],
                                           f.alpha[:,2,0], f.alpha[:,2,1], f.alpha[:,2,2],
                                           f.thole, numpy.int32(f.polar_group), 
                                           numpy.int32(f.connections_12), numpy.int32(f.pointer_connections_12),
                                           numpy.int32(f.connections_13), numpy.int32(f.pointer_connections_13),
                                           dphix_reac, dphiy_reac, dphiz_reac, f.E)

                elif par_reac.GPU==10:
                    GSZ = int(numpy.ceil(float(len(f.q))/par_reac.BSZ)) # CUDA grid size
                    compute_induced_dipole_gpu = kernel.get_function("compute_induced_dipole")

                    dphix_reac_gpu = gpuarray.to_gpu(dphix_reac.astype(par_reac.REAL))
                    dphiy_reac_gpu = gpuarray.to_gpu(dphiy_reac.astype(par_reac.REAL))
                    dphiz_reac_gpu = gpuarray.to_gpu(dphiz_reac.astype(par_reac.REAL))

                    px_pol_gpu = gpuarray.zeros(len(f.q), dtype=par_reac.REAL)
                    py_pol_gpu = gpuarray.zeros(len(f.q), dtype=par_reac.REAL)
                    pz_pol_gpu = gpuarray.zeros(len(f.q), dtype=par_reac.REAL)

                    px_pol_gpu = gpuarray.to_gpu(f.p_pol[:,0].astype(par_reac.REAL))
                    py_pol_gpu = gpuarray.to_gpu(f.p_pol[:,1].astype(par_reac.REAL))
                    pz_pol_gpu = gpuarray.to_gpu(f.p_pol[:,2].astype(par_reac.REAL))

                    compute_induced_dipole_gpu(f.xq_gpu, f.yq_gpu, f.zq_gpu, f.q_gpu,
                                            f.px_gpu, f.py_gpu, f.pz_gpu, 
                                            px_pol_gpu, py_pol_gpu, pz_pol_gpu, 
                                            f.Qxx_gpu, f.Qxy_gpu, f.Qxz_gpu,
                                            f.Qyx_gpu, f.Qyy_gpu, f.Qyz_gpu,
                                            f.Qzx_gpu, f.Qzy_gpu, f.Qzz_gpu,
                                            f.alphaxx_gpu, f.alphaxy_gpu, f.alphaxz_gpu,
                                            f.alphayx_gpu, f.alphayy_gpu, f.alphayz_gpu,
                                            f.alphazx_gpu, f.alphazy_gpu, f.alphazz_gpu, f.thole_gpu, f.polar_group_gpu,
                                            dphix_reac_gpu, dphiy_reac_gpu, dphiz_reac_gpu,
                                            par_reac.REAL(f.E), numpy.int32(len(f.q)), block=(par_reac.BSZ,1,1), grid=(GSZ,1))

                    px_pol_gpu.get(px_pol)
                    py_pol_gpu.get(py_pol)
                    pz_pol_gpu.get(pz_pol)

                f.p_pol[:,0] = px_pol 
                f.p_pol[:,1] = py_pol 
                f.p_pol[:,2] = pz_pol 


def coulombEnergy(f, param, kernel):

    point_energy = numpy.zeros(len(f.q), param.REAL)
    if param.args.polarizable==0: #only point charges
        if param.GPU==0:
            coulomb_direct(f.xq[:,0], f.xq[:,1], f.xq[:,2], f.q, point_energy)

        elif param.GPU==1:

            GSZ = int(numpy.ceil(float(len(f.q))/param.BSZ)) # CUDA grid size
            coulomb_direct_gpu = kernel.get_function("coulomb_direct")
            point_energy_gpu = gpuarray.zeros(len(f.q), dtype=param.REAL)
            coulomb_direct_gpu(f.xq_gpu, f.yq_gpu, f.zq_gpu, f.q_gpu, point_energy_gpu, numpy.int32(len(f.q)), block=(param.BSZ,1,1), grid=(GSZ,1))
            point_energy_gpu.get(point_energy)

    else: # contains multipoles
        if param.GPU==0 or param.GPU==1:
            p_polx = numpy.zeros(len(f.xq))
            p_poly = numpy.zeros(len(f.xq))
            p_polz = numpy.zeros(len(f.xq))
            p_polx[:] = f.p_pol[:,0]
            p_poly[:] = f.p_pol[:,1]
            p_polz[:] = f.p_pol[:,2]
            coulomb_energy_multipole(f.xq[:,0], f.xq[:,1], f.xq[:,2], f.q, 
                                     f.p[:,0], f.p[:,1], f.p[:,2],
                                     p_polx, p_poly, p_polz,
                                     f.Q[:,0,0], f.Q[:,0,1], f.Q[:,0,2],
                                     f.Q[:,1,0], f.Q[:,1,1], f.Q[:,1,2],
                                     f.Q[:,2,0], f.Q[:,2,1], f.Q[:,2,2], 
                                     f.alpha[:,0,0], f.thole, numpy.int32(f.polar_group),
                                     numpy.int32(f.connections_12), numpy.int32(f.pointer_connections_12), 
                                     numpy.int32(f.connections_13), numpy.int32(f.pointer_connections_13), 
                                     param.p12scale, param.p13scale, point_energy)
        elif param.GPU==10:

            GSZ = int(numpy.ceil(float(len(f.q))/param.BSZ)) # CUDA grid size
            coulomb_energy_multipole_gpu = kernel.get_function("coulomb_energy_multipole")
            point_energy_gpu = gpuarray.zeros(len(f.q), dtype=param.REAL)

            px_pol_gpu = gpuarray.to_gpu(f.p_pol[:,0])
            py_pol_gpu = gpuarray.to_gpu(f.p_pol[:,1])
            pz_pol_gpu = gpuarray.to_gpu(f.p_pol[:,2])

#            f.px_gpu = gpuarray.to_gpu(f.p[:,0])
#            f.py_gpu = gpuarray.to_gpu(f.p[:,1])
#            f.pz_gpu = gpuarray.to_gpu(f.p[:,2])

            coulomb_energy_multipole_gpu(f.xq_gpu, f.yq_gpu, f.zq_gpu, f.q_gpu, 
                                         f.px_gpu, f.py_gpu, f.pz_gpu,
                                         px_pol_gpu, py_pol_gpu, pz_pol_gpu,
                                         f.Qxx_gpu, f.Qxy_gpu, f.Qxz_gpu,
                                         f.Qyx_gpu, f.Qyy_gpu, f.Qyz_gpu,
                                         f.Qzx_gpu, f.Qzy_gpu, f.Qzz_gpu,
                                         f.alphaxx_gpu, f.thole_gpu,
                                         point_energy_gpu, numpy.int32(len(f.q)), 
                                         block=(param.BSZ,1,1), grid=(GSZ,1))

            point_energy_gpu.get(point_energy)

    cal2J = 4.184
    C0 = param.qe**2*param.Na*1e-3*1e10/(cal2J*param.E_0)
    Ecoul = numpy.sum(point_energy) * 0.5*C0/(4*pi*f.E)

    return Ecoul

def coulomb_polarizable_dipole(f, param, kernel):
    """  
    Computes polarized dipole component of a collection of polarizabe multipoles

    Inputs:
    ------
        f: class, region in the field array that contains the multipoles
    param: class, parameters related to the surface.
    Returns:
    -------
     None: modifies the f.p_pol member of the class f 
    """
    dipole_diff = 1.
    iteration = 0
    p_pol_prev = numpy.zeros((len(f.xq),3))

    px_pol = numpy.zeros(len(f.xq))
    py_pol = numpy.zeros(len(f.xq))
    pz_pol = numpy.zeros(len(f.xq))

    if param.GPU==1:
        px_pol_gpu = gpuarray.zeros(len(f.q), dtype=param.REAL)
        py_pol_gpu = gpuarray.zeros(len(f.q), dtype=param.REAL)
        pz_pol_gpu = gpuarray.zeros(len(f.q), dtype=param.REAL)

        f.px_gpu = gpuarray.to_gpu(f.p[:,0])
        f.py_gpu = gpuarray.to_gpu(f.p[:,1])
        f.pz_gpu = gpuarray.to_gpu(f.p[:,2])

        GSZ = int(numpy.ceil(float(len(f.q))/param.BSZ)) # CUDA grid size
        compute_induced_dipole_gpu = kernel.get_function("compute_induced_dipole")

    # dummy phi_reac. This is a vacuum calculation
    # then there is no solvent reaction
    dphix_reac = numpy.zeros(len(f.xq))
    dphiy_reac = numpy.zeros(len(f.xq))
    dphiz_reac = numpy.zeros(len(f.xq))

    while dipole_diff>1e-2:
        iteration += 1
        
        if param.GPU==0 or param.GPU==1:
            compute_induced_dipole(f.xq[:,0], f.xq[:,1], f.xq[:,2], f.q, 
                                   f.p[:,0], f.p[:,1], f.p[:,2],
                                   px_pol, py_pol, pz_pol,
                                   f.Q[:,0,0], f.Q[:,0,1], f.Q[:,0,2],
                                   f.Q[:,1,0], f.Q[:,1,1], f.Q[:,1,2],
                                   f.Q[:,2,0], f.Q[:,2,1], f.Q[:,2,2],
                                   f.alpha[:,0,0], f.alpha[:,0,1], f.alpha[:,0,2],
                                   f.alpha[:,1,0], f.alpha[:,1,1], f.alpha[:,1,2],
                                   f.alpha[:,2,0], f.alpha[:,2,1], f.alpha[:,2,2],
                                   f.thole, numpy.int32(f.polar_group), 
                                   numpy.int32(f.connections_12), numpy.int32(f.pointer_connections_12),
                                   numpy.int32(f.connections_13), numpy.int32(f.pointer_connections_13),
                                   dphix_reac, dphiy_reac, dphiz_reac, f.E)
        elif param.GPU==10:

            dphix_reac_gpu = gpuarray.to_gpu(dphix_reac.astype(param.REAL))
            dphiy_reac_gpu = gpuarray.to_gpu(dphiy_reac.astype(param.REAL))
            dphiz_reac_gpu = gpuarray.to_gpu(dphiz_reac.astype(param.REAL))

            compute_induced_dipole_gpu(f.xq_gpu, f.yq_gpu, f.zq_gpu, f.q_gpu,
                                    f.px_gpu, f.py_gpu, f.pz_gpu, 
                                    px_pol_gpu, py_pol_gpu, pz_pol_gpu, 
                                    f.Qxx_gpu, f.Qxy_gpu, f.Qxz_gpu,
                                    f.Qyx_gpu, f.Qyy_gpu, f.Qyz_gpu,
                                    f.Qzx_gpu, f.Qzy_gpu, f.Qzz_gpu,
                                    f.alphaxx_gpu, f.alphaxy_gpu, f.alphaxz_gpu,
                                    f.alphayx_gpu, f.alphayy_gpu, f.alphayz_gpu,
                                    f.alphazx_gpu, f.alphazy_gpu, f.alphazz_gpu, f.thole_gpu, f.polar_group_gpu,
                                    dphix_reac_gpu, dphiy_reac_gpu, dphiz_reac_gpu,
                                    param.REAL(f.E), numpy.int32(len(f.q)), block=(param.BSZ,1,1), grid=(GSZ,1))

            px_pol_gpu.get(px_pol)
            py_pol_gpu.get(py_pol)
            pz_pol_gpu.get(pz_pol)


        f.p_pol[:,0] = px_pol 
        f.p_pol[:,1] = py_pol 
        f.p_pol[:,2] = pz_pol 

        dipole_diff = numpy.sqrt(numpy.sum((numpy.linalg.norm(p_pol_prev-f.p_pol,axis=1))**2)/len(f.p_pol))
        p_pol_prev = f.p_pol.copy()
        print 'Dipole residual in vacuum in iteration %i: %s'%(iteration, dipole_diff)

    print '%i iterations for vacuum induced dipole to converge'%iteration

def calculateEsurf(surf_array, field_array, param, kernel):

    REAL = param.REAL

    par_reac = parameters()
    par_reac = param
    par_reac.threshold = 0.05
    par_reac.P = 7
    par_reac.theta = 0.0
    par_reac.Nm= (par_reac.P+1)*(par_reac.P+2)*(par_reac.P+3)/6

    ind_reac = index_constant()
    computeIndices(par_reac.P, ind_reac)
    precomputeTerms(par_reac.P, ind_reac)

    par_reac.Nk = 13         # Number of Gauss points per side for semi-analytical integrals

    cal2J = 4.184
    C0 = param.qe**2*param.Na*1e-3*1e10/(cal2J*param.E_0)
    E_surf = []

    ff = -1
    for f in param.E_field:
        parent_surf = surf_array[field_array[f].parent[0]]
         
        if parent_surf.surf_type == 'dirichlet_surface':
            ff += 1
            print 'Calculating surface energy around region %i, stored in E_surf[%i]'%(f,ff)
            Esurf_aux = -numpy.sum(-parent_surf.Eout*parent_surf.dphi*parent_surf.phi*parent_surf.Area) 
            E_surf.append(0.5*C0*Esurf_aux)
        
        elif parent_surf.surf_type == 'neumann_surface':
            ff += 1
            print 'Calculating surface energy around region %i, stored in E_surf[%i]'%(f,ff)
            Esurf_aux = numpy.sum(-parent_surf.Eout*parent_surf.dphi*parent_surf.phi*parent_surf.Area) 
            E_surf.append(0.5*C0*Esurf_aux)

    return E_surf
