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

"""
All functions output the analytical solution in kcal/mol
"""
from numpy import *
from scipy import special
from scipy.special import lpmv
from scipy.misc import factorial
from math import gamma
from scipy.linalg import solve

def transformation_matrix(theta,phi):
    M = zeros((3,3))
    M[0,0] = sin(theta)*cos(phi)
    M[0,1] = cos(theta)*cos(phi)
    M[0,2] = -sin(phi)
    M[1,0] = sin(theta)*sin(phi)
    M[1,1] = cos(theta)*sin(phi)
    M[1,2] = cos(phi)
    M[2,0] = cos(theta)
    M[2,1] = -sin(theta)
    M[2,2] = 0

    return M


def sph2cart(spherical_vector, position):

    cartesian_vector = zeros((len(spherical_vector), 3), dtype=complex)
    r_comp      = spherical_vector[:,0]
    theta_comp  = spherical_vector[:,1]
    phi_comp    = spherical_vector[:,2]

    theta = arccos(position[:,2]/sqrt(sum(position**2, axis=1)))
    phi =  arctan2(position[:,1],position[:,0])

    x = r_comp*sin(theta)*cos(phi) + theta_comp*cos(theta)*cos(phi) - phi_comp*sin(phi)
    y = r_comp*sin(theta)*sin(phi) + theta_comp*cos(theta)*sin(phi) + phi_comp*cos(phi)
    z = r_comp*cos(theta) - theta_comp*sin(theta)

    cartesian_vector[:,0] = x[:]
    cartesian_vector[:,1] = y[:]
    cartesian_vector[:,2] = z[:]

    return cartesian_vector 

def sph2cart_vector(spherical_vector, position):

    theta = arccos(position[:,2]/sqrt(sum(position**2, axis=1)))
    phi =  arctan2(position[:,1],position[:,0])

    cartesian_vector = zeros((len(spherical_vector), 3), dtype=complex)

    for i in range(len(spherical_vector)):
        M = transformation_matrix(theta[i],phi[i])
        cartesian_vector[i,:] = dot(M, spherical_vector[i,:])

    return cartesian_vector

def sph2cart_tensor(spherical_tensor, position):

    theta = arccos(position[:,2]/sqrt(sum(position**2, axis=1)))
    phi =  arctan2(position[:,1],position[:,0])

    cartesian_tensor = zeros((len(spherical_tensor), 3, 3), dtype=complex)

    for i in range(len(spherical_tensor)):
        M = transformation_matrix(theta[i],phi[i])
        M2 = dot(M,spherical_tensor[i,:,:])
        cartesian_tensor[i,:,:] = dot(M2, transpose(M))

    return cartesian_tensor

def Ynm_theta_derivative(m, n, azim, zenit, Ynm):

    Ynmp1 = special.sph_harm(m+1,n,azim,zenit)
    Ynmm1 = special.sph_harm(m-1,n,azim,zenit)
    if m==0 and n==0:
        dYnm_dtheta = 0
    elif zenit==0:
        dYnm_dtheta = 0.
    elif m+1<n:
        dYnm_dtheta =  sqrt((n-m)*(n+m+1))*exp(-1j*azim)*Ynmp1 \
                            + m/tan(zenit)*Ynm
    else:
        dYnm_dtheta =  -sqrt((n+m)*(n-m+1))*exp(1j*azim)*Ynmm1 \
                            - m/tan(zenit)*Ynm
    return dYnm_dtheta

def Ynmst_theta_derivative(m, n, azim, zenit, Ynm_st):

    Ynmp1_st = conj(special.sph_harm(m+1,n,azim,zenit))
    Ynmm1_st = conj(special.sph_harm(m-1,n,azim,zenit))

    index = where(abs(zenit)>1e-10)[0] # Indices where theta is nonzero
    dYnmst_dtheta = zeros(len(azim), dtype=complex)
    if m==0 and n==0:
        dYnmst_dtheta = zeros(len(azim), dtype=complex)
    elif m+1<n:
        dYnmst_dtheta[index] =  sqrt((n-m)*(n+m+1))*exp(1j*azim[index])*Ynmp1_st[index] \
                            + m/tan(zenit[index])*Ynm_st[index]
    else:
        dYnmst_dtheta[index] =  -sqrt((n+m)*(n-m+1))*exp(-1j*azim[index])*Ynmm1_st[index] \
                            - m/tan(zenit[index])*Ynm_st[index]
    return dYnmst_dtheta

def Ynmst_theta_2derivative(m, n, azim, zenit, Ynm_st, dYnmst_dtheta):

    Ynmp1_st = conj(special.sph_harm(m+1,n,azim,zenit))
    Ynmm1_st = conj(special.sph_harm(m-1,n,azim,zenit))

    index = where(abs(zenit)>1e-10)[0] # Indices where theta is nonzero
    d2Ynmst_dtheta2 = zeros(len(zenit), dtype=complex)
    if m==0 and n==0:
        dYnmst_dtheta = 0
    elif m+1<n:
        dYnmp1st_dtheta = -sqrt((n+m+1)*(n-m))*exp(-1j*azim[index])*Ynm_st[index] \
                            - (m+1)/tan(zenit[index])*Ynmp1_st[index]
        d2Ynmst_dtheta2[index] =  sqrt((n-m)*(n+m+1))*exp(1j*azim[index])*dYnmp1st_dtheta \
                                + m*(-1/sin(zenit[index])**2*Ynm_st + 1/tan(zenit[index])*dYnmst_dtheta[index])

    else:
        dYnmm1st_dtheta = sqrt((n-m+1)*(n+m))*exp(1j*azim[index])*Ynm_st[index] \
                        + (m-1) * 1/tan(zenit[index])*Ynmm1_st[index]
        d2Ynmst_dtheta2[index] =  -sqrt((n+m)*(n-m+1))*exp(-1j*azim[index])*dYnmm1st_dtheta \
                                - m*(-1/sin(zenit[index])**2*Ynm_st + 1/tan(zenit[index])*dYnmst_dtheta[index])

    return d2Ynmst_dtheta2

def Ynm_theta_2derivative(m, n, azim, zenit, Ynm, dYnm_dtheta):

    Ynmp1 = special.sph_harm(m+1,n,azim,zenit)
    Ynmm1 = special.sph_harm(m-1,n,azim,zenit)

    d2Ynm_dtheta2 = zeros(1, dtype=complex)
    if m==0 and n==0:
        dYnmst_dtheta = 0
    elif abs(zenit)<1e-10:
        dYnmst_dtheta = 0
    elif m+1<n:
        dYnmp1_dtheta = -sqrt((n+m+1)*(n-m))*exp(1j*azim)*Ynm \
                            - (m+1)/tan(zenit)*Ynmp1
        d2Ynm_dtheta2 =  sqrt((n-m)*(n+m+1))*exp(-1j*azim)*dYnmp1_dtheta \
                                + m*(-1/sin(zenit)**2*Ynm + 1/tan(zenit)*dYnm_dtheta)

    else:
        dYnmm1_dtheta = sqrt((n-m+1)*(n+m))*exp(-1j*azim)*Ynm \
                        + (m-1) * 1/tan(zenit)*Ynmm1
        d2Ynm_dtheta2 =  -sqrt((n+m)*(n-m+1))*exp(1j*azim)*dYnmm1_dtheta \
                                - m*(-1/sin(zenit)**2*Ynm + 1/tan(zenit)*dYnm_dtheta)

    return d2Ynm_dtheta2

def vector_grad_rYnmst(m, n, rho, azim, zenit, Ynm_st):

    grad_gradSpherical = zeros((len(azim),3,3), dtype=complex)

    dYnmst_dtheta   = Ynmst_theta_derivative(m, n, azim, zenit, Ynm_st)
    dYnmst_dphi     = -1j*m*Ynm_st
    d2Ynmst_dtheta2 = Ynmst_theta_2derivative(m, n, azim, zenit, Ynm_st, dYnmst_dtheta) 
    d2Ynmst_dphi2   = -m**2*Ynm_st
    d2Ynmst_dthdphi = -1j*m*dYnmst_dtheta

    i_th = where(abs(zenit)>1e-10)[0] # Indices where theta is nonzero
    i_rh = where(rho>1e-16)[0]        # Indices where rho is nonzero
    i_tr = where(logical_and(rho>1e-16,abs(zenit)>1e-10))[0] # Indices where rho and
                                                    # theta are nonzero

    Ar = n*rho**(n-1)*Ynm_st
    dAr_dr  = n*(n-1)*rho**(n-2)*Ynm_st
    dAr_dth = n*rho**(n-1)*dYnmst_dtheta
    dAr_dph = n*rho**(n-1)*dYnmst_dphi

    Ath = rho**(n-1)*dYnmst_dtheta
    dAth_dr  = (n-1)*rho**(n-2) * dYnmst_dtheta
    dAth_dth = rho**(n-1) * d2Ynmst_dtheta2 
    dAth_dph = rho**(n-1) * d2Ynmst_dthdphi

    Aph      = zeros(shape(Ar), dtype=complex)
    dAph_dr  = zeros(shape(Ar), dtype=complex)
    dAph_dth = zeros(shape(Ar), dtype=complex)
    dAph_dph = zeros(shape(Ar), dtype=complex)
    Aph[i_th]      = rho[i_th]**(n-1)/sin(zenit[i_th]) * dYnmst_dphi[i_th]
    dAph_dr[i_th]  = (n-1)*rho[i_th]**(n-2)/sin(zenit[i_th]) * dYnmst_dphi[i_th]
    dAph_dth[i_th] = rho[i_th]**(n-1)* (-cos(zenit[i_th])/sin(zenit[i_th])**2 \
                    * dYnmst_dphi[i_th] + 1/sin(zenit[i_th])*d2Ynmst_dthdphi[i_th])
    dAph_dph[i_th] = rho[i_th]**(n-1)/sin(zenit[i_th]) * d2Ynmst_dphi2[i_th]

    grad_gradSpherical[:,0,0] = dAr_dr
    grad_gradSpherical[:,0,1] = dAth_dr
    grad_gradSpherical[:,0,2] = dAph_dr

    grad_gradSpherical[i_rh,1,0] = (dAr_dth - Ath)[i_rh]/rho[i_rh]

    grad_gradSpherical[i_rh,1,1] = (dAth_dth + Ar)[i_rh]/rho[i_rh]
    grad_gradSpherical[i_rh,1,2] = dAph_dth[i_rh]/rho[i_rh]

    grad_gradSpherical[i_tr,2,0] = (dAr_dph[i_tr]/sin(zenit[i_tr]) - Aph[i_tr])/rho[i_tr]
    grad_gradSpherical[i_tr,2,1] = (dAth_dph[i_tr]/sin(zenit[i_tr]) - Aph[i_tr]/tan(zenit[i_tr]))/rho[i_tr]
    grad_gradSpherical[i_tr,2,2] = (dAph_dph[i_tr]/sin(zenit[i_tr]) + Ar[i_tr] + Ath[i_tr]/tan(zenit[i_tr]))/rho[i_tr]

    return grad_gradSpherical

def vector_grad_rYnm(m, n, rho, azim, zenit, Ynm):

    grad_gradPhi_sph = zeros((1,3,3), dtype=complex)

    dYnm_dtheta   = Ynm_theta_derivative(m, n, azim, zenit, Ynm)
    dYnm_dphi     = 1j*m*Ynm
    d2Ynm_dtheta2 = Ynm_theta_2derivative(m, n, azim, zenit, Ynm, dYnm_dtheta) 
    d2Ynm_dphi2   = -m**2*Ynm
    d2Ynm_dthdphi = 1j*m*dYnm_dtheta

    Ar = n*rho**(n-1)*Ynm
    dAr_dr  = n*(n-1)*rho**(n-2)*Ynm
    dAr_dth = n*rho**(n-1)*dYnm_dtheta
    dAr_dph = n*rho**(n-1)*dYnm_dphi

    Ath = rho**(n-1)*dYnm_dtheta
    dAth_dr  = (n-1)*rho**(n-2) * dYnm_dtheta
    dAth_dth = rho**(n-1) * d2Ynm_dtheta2 
    dAth_dph = rho**(n-1) * d2Ynm_dthdphi

    if abs(sin(zenit))>1e-10:
        Aph = rho**(n-1)/sin(zenit) * dYnm_dphi
        dAph_dr = (n-1)*rho**(n-2)/sin(zenit) * dYnm_dphi
        dAph_dth = rho**(n-1)* (-cos(zenit)/sin(zenit)**2 * dYnm_dphi + 1/sin(zenit)*d2Ynm_dthdphi)
        dAph_dph = rho**(n-1)/sin(zenit) * d2Ynm_dphi2
    else:
        Aph      = 0
        dAph_dr  = 0
        dAph_dth = 0
        dAph_dph = 0

    grad_gradPhi_sph[:,0,0] = dAr_dr
    grad_gradPhi_sph[:,0,1] = dAth_dr
    grad_gradPhi_sph[:,0,2] = dAph_dr

    grad_gradPhi_sph[:,1,0] = (dAr_dth - Ath)/rho
    grad_gradPhi_sph[:,1,1] = (dAth_dth + Ar)/rho
    grad_gradPhi_sph[:,1,2] = dAph_dth/rho

    if abs(sin(zenit))>1e-10:
        grad_gradPhi_sph[0,2,0] = (dAr_dph/sin(zenit) - Aph)/rho
        grad_gradPhi_sph[0,2,1] = (dAth_dph/sin(zenit) - Aph/tan(zenit))/rho
        grad_gradPhi_sph[0,2,2] = (dAph_dph/sin(zenit) + Ar + Ath/tan(zenit))/rho
    else:
        grad_gradPhi_sph[0,2,:] = 0

    return grad_gradPhi_sph

def computeMultipoleMoment(m, n, q, p, Q, xq):

    rho_k   = sqrt(sum(xq**2, axis=1))
    zenit_k = arccos(xq[:,2]/rho_k)
    azim_k  = arctan2(xq[:,1],xq[:,0])

    Ynm_st   = conj(special.sph_harm(m,n,azim_k,zenit_k))
    dYnmst_dtheta = Ynmst_theta_derivative(m,n,azim_k,zenit_k,Ynm_st)
    dYnmst_dphi = -1j*m*Ynm_st

#   Monopole
    monopole = sum(q*rho_k**n*Ynm_st)

#   Dipole
#   p in rectangular coordinates
    
    gradSpherical = zeros((len(xq),3), dtype=complex)
    gradSpherical[:,0] = n*rho_k**(n-1)*Ynm_st
    gradSpherical[:,1] = rho_k**(n-1)*dYnmst_dtheta
    gradSpherical[:,2] = rho_k**(n-1)*dYnmst_dphi/sin(zenit_k)

    gradCartesian = sph2cart_vector(gradSpherical, xq)
        
    dipole = sum(p[:,0]*gradCartesian[:,0]) \
             + sum(p[:,1]*gradCartesian[:,1])  \
             + sum(p[:,2]*gradCartesian[:,2])

#   Quadrupole
#   Q is the traceless quadrupole moment

    grad_gradSpherical = vector_grad_rYnmst(m, n, rho_k, azim_k, zenit_k, Ynm_st)
    grad_gradCartesian = sph2cart_tensor(grad_gradSpherical, xq)
    quadrupole = sum(Q[:,0,0]*grad_gradCartesian[:,0,0]) \
               + sum(Q[:,0,1]*grad_gradCartesian[:,0,1]) \
               + sum(Q[:,0,2]*grad_gradCartesian[:,0,2]) \
               + sum(Q[:,1,0]*grad_gradCartesian[:,1,0]) \
               + sum(Q[:,1,1]*grad_gradCartesian[:,1,1]) \
               + sum(Q[:,1,2]*grad_gradCartesian[:,1,2]) \
               + sum(Q[:,2,0]*grad_gradCartesian[:,2,0]) \
               + sum(Q[:,2,1]*grad_gradCartesian[:,2,1]) \
               + sum(Q[:,2,2]*grad_gradCartesian[:,2,2]) 
    
    return monopole + dipole + quadrupole/6 # divided by 6: see modified Kirkwood Part 2b

def computeYnm_derivatives(m,n,xq_k):

    rho = sqrt(sum(xq_k**2))
    zenit = arccos(xq_k[2]/rho)
    azim  = arctan2(xq_k[1],xq_k[0])

    Ynm = special.sph_harm(m,n,azim,zenit)
    dYnm_dtheta = Ynm_theta_derivative(m, n, azim, zenit, Ynm)
    dYnm_dphi = 1j*m*Ynm

    gradPhi_sph = zeros((1,3), dtype=complex)
    gradPhi_sph[:,0] = n*rho**(n-1)*Ynm
    gradPhi_sph[:,1] = rho**(n-1)*dYnm_dtheta
    gradPhi_sph[:,2] = rho**(n-1)*dYnm_dphi/sin(zenit)
    gradPhi_cart = sph2cart_vector(gradPhi_sph, array([xq_k]))

    grad_gradPhi_sph = vector_grad_rYnm(m, n, rho, azim, zenit, Ynm)
    grad_gradPhi_cart = sph2cart_tensor(grad_gradPhi_sph, array([xq_k]))

    return rho**n*Ynm, gradPhi_cart, grad_gradPhi_cart

def coulomb_potential(q, p, Q, xq, E):
    """
    Computes the electrostatic potential due to a point monopole, dipole, 
    quadrupole distribution at the position of the multipoles.
    See equation 29 of amoeba bem document. 
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        phi: electrostatic potential at the position of the multipoles
    """

    phi = zeros(len(xq))
    T2  = zeros((len(xq),3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = sqrt(sum(Ri*Ri, axis=1))

        for j in where(Rnorm>1e-10)[0]: #remove singularity
            T0 = 1/Rnorm[j]
            T1 = Ri[j,:]/Rnorm[j]**3
            T2[j,:,:] = ones((3,3))*Ri[j,:]*transpose(ones((3,3))*Ri[j,:])/Rnorm[j]**5

            phi[i] += q[j]*T0 + sum(T1[:]*p[j,:]) + 0.5*sum(sum(T2[j,:,:]*Q[j,:,:],axis=1),axis=0)

    phi /= (4*pi*E)

    return phi

def coulomb_field(q, p, Q, xq, E):
    """
    Computes the electric field due to a point monopole, dipole, quadrupole distribution
    at the position of the multipoles. The field is defined as E=-nabla*phi.
    See equation 52 of kirkwood multipole, and Equation 30 of amoeba bem document.
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        Efield: electric field at the position of the multipoles
    """
    Efield = zeros((len(xq),3))
    T0 = zeros((len(xq),3))
    T1 = zeros((len(xq),3,3))
    T2 = zeros((len(xq),3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = sqrt(sum(Ri*Ri, axis=1))


        for j in where(Rnorm>1e-10)[0]: #remove singularity
            T0[j,:]   = -Ri[j,:]/Rnorm[j]**3
            T1[j,:,:] = identity(3)/Rnorm[j]**3 - 3*ones((3,3))*Ri[j,:]*transpose(ones((3,3))*Ri[j,:])/Rnorm[j]**5

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = ones((3,3))*Ri[j,:]*transpose(ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= -5/Rnorm[j]**7

            for k in range(3):
                aux[:,:,k] += identity(3)*Ri[j,k]/Rnorm[j]**5
            for k in range(3):
                aux[:,k,:] += identity(3)*Ri[j,k]/Rnorm[j]**5

            T2[j,:,:,:] = aux

            for k in range(3):
                Efield[i,k] += T0[j,k]*q[j] + sum(T1[j,k,:]*p[j,:]) + 0.5*sum(sum(T2[j,k,:,:]*Q[j,:,:],axis=1),axis=0)

    Efield /= -(4*pi*E)
    return Efield

def coulomb_ddpotential(q, p, Q, xq, E):
    """
    Computes the second derivative of the electrostatic potential due 
    to a point monopole, dipole, and quadrupole distribution at the 
    position of the multipoles. See equation 29 and 43 of amoeba bem document. 

    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        ddphi: second derivative of electrostatic potential at the 
                position of the multipoles
    """
    ddphi = zeros((len(xq),3,3))
    T0 = zeros((len(xq),3,3))
    T1 = zeros((len(xq),3,3,3))
    T2 = zeros((len(xq),3,3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = sqrt(sum(Ri*Ri, axis=1))

        for j in where(Rnorm>1e-10)[0]: #remove singularity
            T0[j,:,:] = -identity(3)/Rnorm[j]**3 + 3*ones((3,3))*Ri[j,:]*transpose(ones((3,3))*Ri[j,:])/Rnorm[j]**5

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = ones((3,3))*Ri[j,:]*transpose(ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= 15/Rnorm[j]**7

            for k in range(3):
                aux[:,:,k] -= 3*identity(3)*Ri[j,k]/Rnorm[j]**5
                aux[:,k,:] -= 3*identity(3)*Ri[j,k]/Rnorm[j]**5
                aux[k,:,:] -= 3*identity(3)*Ri[j,k]/Rnorm[j]**5

            T1[j,:,:,:] = aux

            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            dkl = (k==l)*1.0
                            dkm = (k==m)*1.0
                            dkn = (k==n)*1.0
                            dlm = (l==m)*1.0
                            dln = (l==n)*1.0

                            T2[j,k,l,m,n] = -7*Ri[j,k]*Ri[j,l]*Ri[j,m]*Ri[j,n]/Rnorm[j]**2  \
                                           + Ri[j,m]*Ri[j,n]*dkl + Ri[j,l]*Ri[j,n]*dkm      \
                                           + Ri[j,m]*Ri[j,l]*dkn + Ri[j,k]*Ri[j,n]*dlm      \
                                           + Ri[j,m]*Ri[j,k]*dln
            T2 *= -5/Rnorm[j]**7

            for k in range(3):
                for l in range(3):
                    ddphi[i,k,l] += T0[j,k,l]*q[j] + sum(T1[j,k,l,:]*p[j,:]) + 0.5*sum(sum(T2[j,k,l,:,:]*Q[j,:,:],axis=1),axis=0)
    
    ddphi /= (4*pi*E)

    return ddphi
 
def coulomb_polarizable_dipole(q, p_per, Q, alpha, xq, E):
    """
    Computes polarized dipole component of a collection of polarizabe multipoles
    Used in eq 56 of kirkwood multipole

    Inputs:
    ------
        q: array size N with charges of multipoles
        p_per: array size (Nx3) with permanent dipoles of multipoles
        Q: array size (Nx3x3) with quadrupoles of multipoles
        alpha: array size (Nqx3x3) with polarizability of dipoles (considered as a tensor)
        xq: array size Nx3 with positions of multipoles
        E : float, dielectric constant
    Returns:
    -------
        p_pol: array size (Nx3) with polarizable component of dipoles
        Efield: array size (Nx3) with electrostatic field that polarized the multipoles 
    """
    p_pol      = zeros((len(xq),3))
    dipole_diff= 1. 
    p_pol_prev = ones((len(xq),3))

    iteration = 0
    while dipole_diff>1e-10:
        iteration += 1
        p_tot = p_per + p_pol
    
        Efield = coulomb_field(q, p_tot, Q, xq, E)
        
        for k in range(len(q)):
            p_pol[k] = dot(alpha[k],Efield[k])

        dipole_diff = sqrt(sum((linalg.norm(p_pol_prev-p_pol,axis=1))**2)/len(p_pol))
        p_pol_prev = p_pol.copy()

    return p_pol, Efield

def coulomb_energy(q, p, Q, xq, E):
    """
    Computes the Coulomb energy from a collection of point
    multipoles, according to equation 38 of amoeba bem document.
    
    Inputs:
    ------ 
        q: array size N with charges of multipoles
        p: array size (Nx3) with dipoles of multipoles
        Q: array size (Nx3x3) with quadrupoles of multipoles
        xq: array size Nx3 with positions of multipoles
        E : float, dielectric constant
    Outputs:
    -------
        E_coul: (float) free energy  
    """
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    phi   = coulomb_potential(q, p, Q, xq, E)
    dphi  = -1*coulomb_field(q, p, Q, xq, E)
    ddphi = coulomb_ddpotential(q, p, Q, xq, E)

    cons = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E_coul = 0.5*cons*(sum(q*phi) + sum(sum(p*dphi,axis=1)) + sum(sum(sum(Q*ddphi,axis=2),axis=1))/6)

    return E_coul 

def polarization_energy(p_pol_diss, p_pol_vac, Epol_diss, Epol_vac):
    """
    Computes the polarization energy according to Eq 63 in kirkwood multipole

    Inputs:
    ------- 
        p_pol_diss: array size (Nx3) with polarized component of dipole in 
                    dissolved state
        Epol_diss: array size (Nx3) with total field that polarizes the multipoles in 
                    dissolved state
        p_pol_vac : array size (Nx3) with polarized component of dipole in 
                    vacuum state 
        Epol_vac : array size (Nx3) with total field that polarizes the multipoles in 
                    vacuum state
    Returns:
    -------
        polarization_energy: the energy to polarize multipoles from p_pol_vac to p_pol_diss 
    """
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 
    cons = qe**2*Na*1e-3*1e10/(cal2J*E_0)

    polarization_energy = 0.5*cons*sum(sum(p_pol_diss*Epol_diss, axis=1) - sum(p_pol_vac*Epol_vac, axis=1))

    return polarization_energy

def solvation_energy_polarizable(q, p, Q, alpha, xq, E_1, E_2, kappa, R, a, N):
    """
    Computes the total free energy required for the solvation of a spherical cavity
    with a random distribution of point multipoles with a polarizable dipole.
    See Equation 54 of kirkwood_multipole and Equation 36 of amoeba_bem.
    The energy has four components: solvent, coulomb in dissolved state, coulomb
    in vacuum and polarization energy.
    This function calls: 
            - an_multipole_polarizable for the solvent contribution.
            - coulomb_energy twice: once for the coulomb energy in dissolved
                state and once for the coulomb energy in vacuum state.
            - polarization_energy: for the polarization energy.
    
    Inputs:
    ------
        q: array size N with charges of multipoles
        p: array size (Nqx3) with dipoles of multipoles
        Q: array size (Nqx3x3) with quadrupoles of multipoles
        alpha: array size (Nqx3x3) with polarizability of dipoles (considered as a tensor)
        x_q: array size (Nqx3) with position of multipoles
        E_1: (float) dielectric constant inside cavity
        E_1: (float) dielectric constant outside cavity
        kappa: (float) inverse of Debye length
        R: (float) radius of spherical cavity
        a: (float) radius of Stern layer
        N: (int) number of terms in spherical harmonic expansion

    Returns:
    -------
        solvation_energy: (float) Total free energy to dissolve a spherical
            molecule with polarizable dipoles.
    """

    solvent_contribution, Epol_diss, p_pol_diss = \
            an_multipole_polarizable(q, p, Q, alpha, xq, E_1, E_2, kappa, R, a, N)

    p_diss_tot = p + p_pol_diss
    coulomb_dissolved = coulomb_energy(q, p_diss_tot, Q, xq, E_1)

    p_pol_vac, Epol_vac = coulomb_polarizable_dipole(q, p, Q, alpha, xq, E_1)
    p_vac_tot = p + p_pol_vac
    coulomb_vacuum = coulomb_energy(q, p_vac_tot, Q, xq, E_1)

    pol_energy = polarization_energy(p_pol_diss, p_pol_vac, Epol_diss, Epol_vac)
    
    solvation_energy = solvent_contribution + coulomb_dissolved - coulomb_vacuum + pol_energy
    print 'solvent: %f\ncoulomb diss: %f\ncoulomb vac: %f\npolarization: %f'%(solvent_contribution,\
                                                        coulomb_dissolved, coulomb_vacuum, pol_energy)

    return solvation_energy

def an_multipole_polarizable(q, p, Q, alpha, xq, E_1, E_2, kappa, R, a, N):
    """
    Computes the solvent contribution to free energy for a spherical cavity
    with a random distribution of point multipoles with a polarizable dipole.
    This definition is according to Equation 38 and 39 of amoeba bem document.
    
    Inputs:
    ------
        q: array size N with charges of multipoles
        p: array size (Nqx3) with dipoles of multipoles
        Q: array size (Nqx3x3) with quadrupoles of multipoles
        alpha: array size (Nqx3x3) with polarizability of dipoles (considered as a tensor)
        x_q: array size (Nqx3) with position of multipoles
        E_1: (float) dielectric constant inside cavity
        E_1: (float) dielectric constant outside cavity
        kappa: (float) inverse of Debye length
        R: (float) radius of spherical cavity
        a: (float) radius of Stern layer
        N: (int) number of terms in spherical harmonic expansion

    Returns:
    -------
        E_P: (float) Solvent contribution to free energy
        Epol: array of size (Nx3) with total field at multipole 
                sites, which polarizes the multipoles (used to 
                obtain the field required to compute the polarization energy)
        p_pol: array of size (Nx3) with polarized component of multipoles
    """
        
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 
    dipole_diff = 1.

    PHI  = zeros(len(q))
    DPHI = zeros((len(q),3))
    DDPHI = zeros((len(q),3,3))
    p_pol = zeros((len(q),3))
    p_pol_prev = ones((len(q),3))
    p_tot = p.copy()

    iterations = 0
    while dipole_diff>1e-10:
    
        coul_field = coulomb_field(q, p_tot, Q, xq, E_1)
        Epol = coul_field - DPHI
        for K in range(len(q)):
            p_pol[K] = dot(alpha[K],Epol[K])
        
        p_tot = p + p_pol

        for K in range(len(q)):

            phi = 0.+0.*1j
            dphi = zeros((1,3), dtype=complex)
            ddphi = zeros((3,3), dtype=complex)
            for n in range(N):
                for m in range(-n,n+1):

                    Enm = computeMultipoleMoment(m, n, q, p_tot, Q, xq) 
                    
                    C0 = 1/(E_1*R**(2*n+1))*(E_1-E_2)*(n+1)/(E_1*n+E_2*(n+1))
                    C1 = 1/(E_2*a**(2*n+1)) * (2*n+1)/(2*n-1) * (E_2/((n+1)*E_2+n*E_1))**2
                    C2 = (kappa*a)**2*get_K(kappa*a,n-1)/(get_K(kappa*a,n+1) + \
                        n*(E_2-E_1)/((n+1)*E_2+n*E_1)*(R/a)**(2*n+1)*(kappa*a)**2*get_K(kappa*a,n-1)/((2*n-1)*(2*n+1)))

                    if n==0 and m==0:
                        Bnm = Enm/(R)*(1/E_2-1/E_1) - Enm*kappa*a/(E_2*a*(1+kappa*a))
                    else:
                        Bnm = (C0 - C1*C2) * Enm


                    rhoYnm, gradPhi, grad_gradPhi = computeYnm_derivatives(m, n, xq[K])
#                    if n==(N-1):
#                        print n,m
#                        print grad_gradPhi
                    
                    C3 = 4*pi/(2*n+1)
                    phi += Bnm * rhoYnm * C3
                    dphi += Bnm * gradPhi * C3
                    ddphi += Bnm * grad_gradPhi[0] * C3
#                    print n,m,phi,dphi
#                    print ddphi
            
            PHI[K]   = real(phi)/(4*pi)
            DPHI[K]  = real(dphi)/(4*pi)
            DDPHI[K] = real(ddphi)/(4*pi)

        iterations += 1

        dipole_diff = sqrt(sum((linalg.norm(p_pol_prev-p_pol,axis=1))**2)/len(p_pol))
        p_pol_prev = p_pol.copy()

#       print iterations
    cons = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E_P = 0.5*cons*(sum(q*PHI) + sum(sum(p_tot*DPHI,axis=1)) + sum(sum(sum(Q*DDPHI,axis=2),axis=1))/6)

    return E_P, Epol, p_pol

        
def an_multipole(q, p, Q, xq, E_1, E_2, R, N):
        
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    PHI  = zeros(len(q))
    DPHI = zeros((len(q),3))
    DDPHI = zeros((len(q),3,3))
    for K in range(len(q)):
        rho = sqrt(sum(xq[K]**2))
        zenit = arccos(xq[K,2]/rho)
        azim  = arctan2(xq[K,1],xq[K,0])

        phi = 0.+0.*1j
        dphi = zeros((1,3), dtype=complex)
        ddphi = zeros((3,3), dtype=complex)
        for n in range(N):
            for m in range(-n,n+1):

                Ynm = special.sph_harm(m,n,azim,zenit)
                dYnm_dtheta = Ynm_theta_derivative(m, n, azim, zenit, Ynm)
                dYnm_dphi = 1j*m*Ynm

                gradPhi_sph = zeros((1,3), dtype=complex)
                gradPhi_sph[:,0] = n*rho**(n-1)*Ynm
                gradPhi_sph[:,1] = rho**(n-1)*dYnm_dtheta
                gradPhi_sph[:,2] = rho**(n-1)*dYnm_dphi/sin(zenit)
                gradPhi_cart = sph2cart_vector(gradPhi_sph, array([xq[K]]))

                grad_gradPhi_sph = vector_grad_rYnm(m, n, rho, azim, zenit, Ynm)
                grad_gradPhi_cart = sph2cart_tensor(grad_gradPhi_sph, array([xq[K]]))

                C0 = 1/(E_1*E_0*R**(2*n+1))*(E_1-E_2)*(n+1)/(E_1*n+E_2*(n+1))

                rho_k   = sqrt(sum(xq**2, axis=1))
                zenit_k = arccos(xq[:,2]/rho_k)
                azim_k  = arctan2(xq[:,1],xq[:,0])

                Ynm_st   = conj(special.sph_harm(m,n,azim_k,zenit_k))
                dYnmst_dtheta = Ynmst_theta_derivative(m,n,azim_k,zenit_k,Ynm_st)
                dYnmst_dphi = -1j*m*Ynm_st

#               Monopole
                monopole = sum(q*rho_k**n*Ynm_st)

#               Dipole
#               p in rectangular coordinates
                
                gradSpherical = zeros((len(xq),3), dtype=complex)
                gradSpherical[:,0] = n*rho_k**(n-1)*Ynm_st
                gradSpherical[:,1] = rho_k**(n-1)*dYnmst_dtheta
                gradSpherical[:,2] = rho_k**(n-1)*dYnmst_dphi/sin(zenit_k)

                gradCartesian = sph2cart_vector(gradSpherical, xq)
                    
                dipole = sum(p[:,0]*gradCartesian[:,0]) \
                         + sum(p[:,1]*gradCartesian[:,1])  \
                         + sum(p[:,2]*gradCartesian[:,2])

#               Quadrupole
#               Q is the traceless quadrupole moment

                grad_gradSpherical = vector_grad_rYnmst(m, n, rho_k, azim_k, zenit_k, Ynm_st)
                grad_gradCartesian = sph2cart_tensor(grad_gradSpherical, xq)
                quadrupole = sum(Q[:,0,0]*grad_gradCartesian[:,0,0]) \
                           + sum(Q[:,0,1]*grad_gradCartesian[:,0,1]) \
                           + sum(Q[:,0,2]*grad_gradCartesian[:,0,2]) \
                           + sum(Q[:,1,0]*grad_gradCartesian[:,1,0]) \
                           + sum(Q[:,1,1]*grad_gradCartesian[:,1,1]) \
                           + sum(Q[:,1,2]*grad_gradCartesian[:,1,2]) \
                           + sum(Q[:,2,0]*grad_gradCartesian[:,2,0]) \
                           + sum(Q[:,2,1]*grad_gradCartesian[:,2,1]) \
                           + sum(Q[:,2,2]*grad_gradCartesian[:,2,2]) 
            
                Enm = monopole + dipole + quadrupole/6 # divided by 6: see modified Kirkwood Part 2b

                
                C2 = (kappa*a)**2*get_K(kappa*a,n-1)/(get_K(kappa*a,n+1) + \
                    n*(E_2-E_1)/((n+1)*E_2+n*E_1)*(R/a)**(2*n+1)*(kappa*a)**2*get_K(kappa*a,n-1)/((2*n-1)*(2*n+1)))
                C1 = 1/(E_2*E_0*a**(2*n+1)) * (2*n+1)/(2*n-1) * (E_2/((n+1)*E_2+n*E_1))**2

                if n==0 and m==0:
                    Bnm = Enm/(E_0*R)*(1/E_2-1/E_1) - Enm*kappa*a/(E_0*E_2*a*(1+kappa*a))
                else:
                    Bnm = (C0 - C1*C2) * Enm

                C3 = 4*pi/(2*n+1)
                phi += Bnm * Ynm * rho**n * C3
                dphi += Bnm * gradPhi_cart * C3
                ddphi += Bnm * grad_gradPhi_cart[0] * C3
        
        PHI[K]   = real(phi)/(4*pi)
        DPHI[K]  = real(dphi)/(4*pi)
        DDPHI[K] = real(ddphi)/(4*pi)

    cons = qe**2*Na*1e-3*1e10/(cal2J)
    
    E_P = 0.5*cons*(sum(q*PHI) + sum(sum(p*DPHI,axis=1)) + sum(sum(sum(Q*DDPHI,axis=2),axis=1))/6)

    return E_P



def an_multipole_2(q, p, Q, xq, E_1, E_2, R, N):
        
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    PHI  = zeros(len(q))
    DPHI = zeros((len(q),3))
    for K in range(len(q)):
        rho = sqrt(sum(xq[K]**2))
        zenit = arccos(xq[K,2]/rho)
        azim  = arctan2(xq[K,1],xq[K,0])

        phi = 0.+0.*1j
        dphi = zeros((1,3), dtype=complex)
        for n in range(N):
            for m in range(-n,n+1):
                Ynm   = special.sph_harm(m,n,azim,zenit)
                Ynmp1 = special.sph_harm(m+1,n,azim,zenit)
                Ynmm1 = special.sph_harm(m-1,n,azim,zenit)
                dYnm_dtheta = zeros(len(xq), dtype=complex)
                if m==0 and n==0:
                    dYnm_dtheta = 0
                elif zenit==0:
                    dYnm_dtheta = 0.
                elif m+1<n:
                    dYnm_dtheta =  sqrt((n-m)*(n+m+1))*exp(-1j*azim)*Ynmp1 \
                                        + m/tan(zenit)*Ynm
                else:
                    dYnm_dtheta =  -sqrt((n+m)*(n-m+1))*exp(1j*azim)*Ynmm1 \
                                        - m/tan(zenit)*Ynm

                dYnm_dphi = 1j*m*Ynm

                gradPhi_sph = zeros((len(xq),3), dtype=complex)
                gradPhi_sph[:,0] = n*rho**(n-1)*Ynm
                gradPhi_sph[:,1] = rho**(n-1)*dYnm_dtheta
                gradPhi_sph[:,2] = rho**(n-1)*dYnm_dphi/sin(zenit)

                gradPhi_cart = sph2cart(gradPhi_sph, array([xq[K]]))

                rho_k   = sqrt(sum(xq**2, axis=1))
                zenit_k = arccos(xq[:,2]/rho_k)
                azim_k  = arctan2(xq[:,1],xq[:,0])

                Ynm_st   = conj(special.sph_harm(m,n,azim_k,zenit_k))
                Ynmp1_st = conj(special.sph_harm(m+1,n,azim_k,zenit_k))
                Ynmm1_st = conj(special.sph_harm(m-1,n,azim_k,zenit_k))

#               Monopole
                monopole = sum(q*rho_k**n*Ynm_st)

#               Dipole
#               p in rectangular coordinates
                
                index = where(abs(zenit_k)>1e-10)[0] # Indices where theta is nonzero
                dYnmst_dtheta = zeros(len(xq), dtype=complex)
                if m==0 and n==0:
                    dYnmst_dtheta = 0
                elif m+1<n:
                    dYnmst_dtheta[index] =  sqrt((n-m)*(n+m+1))*exp(1j*azim_k[index])*Ynmp1_st[index] \
                                        + m/tan(zenit_k[index])*Ynm_st[index]
                else:
                    dYnmst_dtheta[index] =  -sqrt((n+m)*(n-m+1))*exp(-1j*azim_k[index])*Ynmm1_st[index] \
                                        - m/tan(zenit_k[index])*Ynm_st[index]

                dYnmst_dphi = -1j*m*Ynm_st

                gradSpherical = zeros((len(xq),3), dtype=complex)
                gradSpherical[:,0] = n*rho_k**(n-1)*Ynm_st
                gradSpherical[:,1] = rho_k**(n-1)*dYnmst_dtheta
                gradSpherical[:,2] = rho_k**(n-1)*dYnmst_dphi/sin(zenit_k)

                gradCartesian = sph2cart(gradSpherical, xq)
                    
                dipole = -sum(p[:,0]*gradCartesian[:,0]) \
                         - sum(p[:,1]*gradCartesian[:,1])  \
                         - sum(p[:,2]*gradCartesian[:,2])

                qnm = monopole + dipole

                K1   = special.kv(n+0.5, kappa*R)
                K1p1 = special.kv(n+1.5, kappa*R)
                K1p = (n+0.5)/(kappa*R)*K1 - K1p1

                k = special.kv(n+0.5, kappa*R)*sqrt(pi/(2*kappa*R))
                kp = -sqrt(pi/2)*1/(2*(kappa*R)**(3/2.))*special.kv(n+0.5, kappa*R) + sqrt(pi/(2*kappa*R))*K1p
                C0 = (E_1*(n+1)*k + E_2*R*kappa*kp)/(E_1*n*k - E_2*kappa*R*kp)

                Bnm = 4*pi/(E_1*(2*n+1)) * qnm/R**(2*n+1) * C0

                phi += Bnm * Ynm * rho**n 
                dphi += Bnm * gradPhi_cart
        
        PHI[K] = real(phi)/(4*pi)
        DPHI[K] = abs(dphi)/(4*pi)

    cons = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E_P = 0.5*cons*(sum(q*PHI) + sum(sum(p*DPHI,axis=1)))

    return E_P

def an_spherical(q, xq, E_1, E_2, R, N):
        
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    PHI = zeros(len(q))
    for K in range(len(q)):
        rho = sqrt(sum(xq[K]**2))
        zenit = arccos(xq[K,2]/rho)
        azim  = arctan2(xq[K,1],xq[K,0])

        phi = 0.+0.*1j
        for n in range(N):
            for m in range(-n,n+1):
                sph1 = special.sph_harm(m,n,azim,zenit)
                C0 = 1/(E_1*E_0*R**(2*n+1))*(E_1-E_2)*(n+1)/(E_1*n+E_2*(n+1))

                rho_k   = sqrt(sum(xq**2, axis=1))
                zenit_k = arccos(xq[:,2]/rho_k)
                azim_k  = arctan2(xq[:,1],xq[:,0])
                sph2 = conj(special.sph_harm(m,n,azim_k,zenit_k))

                Enm = sum(q*rho_k**n*sph2) # qlm in Jackson

                C2 = (kappa*a)**2*get_K(kappa*a,n-1)/(get_K(kappa*a,n+1) + 
                        n*(E_2-E_1)/((n+1)*E_2+n*E_1)*(R/a)**(2*n+1)*(kappa*a)**2*get_K(kappa*a,n-1)/((2*n-1)*(2*n+1)))
                C1 = 1/(E_2*E_0*a**(2*n+1)) * (2*n+1)/(2*n-1) * (E_2/((n+1)*E_2+n*E_1))**2

                if n==0 and m==0:
                    Bnm = Enm/(E_0*R)*(1/E_2-1/E_1) - Enm*kappa*a/(E_0*E_2*a*(1+kappa*a))
                else:
                    Bnm = (C0 - C1*C2) * Enm

                C3 = 4*pi/(2*n+1)
                phi += Bnm * sph1 * rho**n * C3
        
        PHI[K] = real(phi)/(4*pi)

    C0 = qe**2*Na*1e-3*1e10/(cal2J)
    E_P = 0.5*C0*sum(q*PHI)

    return E_P

def get_K(x,n):

    K = 0.
    n_fact = factorial(n)
    n_fact2 = factorial(2*n)
    for s in range(n+1):
        K += 2**s*n_fact*factorial(2*n-s)/(factorial(s)*n_fact2*factorial(n-s)) * x**s

    return K

def an_P(q, xq, E_1, E_2, R, kappa, a, N):

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    PHI = zeros(len(q))
    for K in range(len(q)):
        rho = sqrt(sum(xq[K]**2))
        zenit = arccos(xq[K,2]/rho)
        azim  = arctan2(xq[K,1],xq[K,0])

        phi = 0.+0.*1j
        for n in range(N):
            for m in range(-n,n+1):
                P1 = lpmv(abs(m),n,cos(zenit))

                Enm = 0.
                for k in range(len(q)):
                    rho_k   = sqrt(sum(xq[k]**2))
                    zenit_k = arccos(xq[k,2]/rho_k)
                    azim_k  = arctan2(xq[k,1],xq[k,0])
                    P2 = lpmv(abs(m),n,cos(zenit_k))

                    Enm += q[k]*rho_k**n*factorial(n-abs(m))/factorial(n+abs(m))*P2*exp(-1j*m*azim_k)
    
                C2 = (kappa*a)**2*get_K(kappa*a,n-1)/(get_K(kappa*a,n+1) + 
                        n*(E_2-E_1)/((n+1)*E_2+n*E_1)*(R/a)**(2*n+1)*(kappa*a)**2*get_K(kappa*a,n-1)/((2*n-1)*(2*n+1)))
                C1 = Enm/(E_2*E_0*a**(2*n+1)) * (2*n+1)/(2*n-1) * (E_2/((n+1)*E_2+n*E_1))**2

                if n==0 and m==0:
                    Bnm = Enm/(E_0*R)*(1/E_2-1/E_1) - Enm*kappa*a/(E_0*E_2*a*(1+kappa*a))
                else:
                    Bnm = 1./(E_1*E_0*R**(2*n+1)) * (E_1-E_2)*(n+1)/(E_1*n+E_2*(n+1)) * Enm - C1*C2

                phi += Bnm*rho**n*P1*exp(1j*m*azim)

        PHI[K] = real(phi)/(4*pi)

    C0 = qe**2*Na*1e-3*1e10/(cal2J)
    E_P = 0.5*C0*sum(q*PHI)

    return E_P


def an_spherical2(q, xq, E_1, E_2, R, kappa, a, N):

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    PHI = zeros(len(q))
    for K in range(len(q)):
        rho = sqrt(sum(xq[K]**2))
        zenit = arccos(xq[K,2]/rho)
        azim  = arctan2(xq[K,1],xq[K,0])

        phi = 0.+0.*1j
        for n in range(N):
            for m in range(-n,n+1):
                P1 = lpmv(abs(m),n,cos(zenit))

                Enm = 0.
                for k in range(len(q)):
                    rho_k   = sqrt(sum(xq[k]**2))
                    zenit_k = arccos(xq[k,2]/rho_k)
                    azim_k  = arctan2(xq[k,1],xq[k,0])
                    P2 = lpmv(abs(m),n,cos(zenit_k))

#                    Enm += q[k]*rho_k**n*factorial(n-abs(m))/factorial(n+abs(m))*P2*exp(-1j*m*azim_k)
                    Enm += q[k]*rho_k**n*conjugate(special.sph_harm(m, n, azim_k, zenit_k))
    
                C2 = (kappa*a)**2*get_K(kappa*a,n-1)/(get_K(kappa*a,n+1) + 
                        n*(E_2-E_1)/((n+1)*E_2+n*E_1)*(R/a)**(2*n+1)*(kappa*a)**2*get_K(kappa*a,n-1)/((2*n-1)*(2*n+1)))
                C1 = Enm/(E_2*E_0*a**(2*n+1)) * (2*n+1)/(2*n-1) * (E_2/((n+1)*E_2+n*E_1))**2

                if n==0 and m==0:
                    Bnm = (Enm/(E_0*R)*(1/E_2-1/E_1) - Enm*kappa*a/(E_0*E_2*a*(1+kappa*a)))*4*pi/(2*n+1)
                else:
                    Bnm = (1./(E_1*E_0*R**(2*n+1)) * (E_1-E_2)*(n+1)/(E_1*n+E_2*(n+1)) * Enm  - C1*C2)* 4*pi/(2*n+1)

#                phi += Bnm*rho**n*P1*exp(1j*m*azim)
                phi += Bnm*rho**n*special.sph_harm(m, n, azim, zenit)

        PHI[K] = real(phi)/(4*pi)

    C0 = qe**2*Na*1e-3*1e10/(cal2J)
    E_P = 0.5*C0*sum(q*PHI)

    return E_P

def two_sphere_KimSong(a, R, kappa, E_1, E_2, q):

    E_hat = E_2/E_1
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    C0 = -q/(a*a*E_hat*kappa*E_1)
    k0a = exp(-kappa*a)/(kappa*a)
    k0R = exp(-kappa*R)/(kappa*R)
    k1a = -k0a - k0a/(kappa*a)
    i0 = sinh(kappa*a)/(kappa*a)
    i1 = cosh(kappa*a)/(kappa*a) - i0/(kappa*a)

    CC0 = qe**2*Na*1e-3*1e10/(cal2J*E_0*4*pi)

    Einter = 0.5*q*C0*CC0*( (k0a+k0R*i0)/(k1a+k0R*i1) - k0a/k1a)
    E1sphere = 0.5*q*C0*CC0*(k0a/k1a) - 0.5*CC0*q**2/(a*E_1)
    E2sphere = 0.5*q*C0*CC0*(k0a+k0R*i0)/(k1a+k0R*i1) - 0.5*CC0*q**2/(a*E_1)

    return Einter, E1sphere, E2sphere

def two_sphere(a, R, kappa, E_1, E_2, q):

    N = 20 # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    index2 = arange(N+1, dtype=float) + 0.5
    index  = index2[0:-1]

    K1 = special.kv(index2, kappa*a)
    K1p = index/(kappa*a)*K1[0:-1] - K1[1:]
    
    k1 = special.kv(index, kappa*a)*sqrt(pi/(2*kappa*a))
    k1p = -sqrt(pi/2)*1/(2*(kappa*a)**(3/2.))*special.kv(index, kappa*a) + sqrt(pi/(2*kappa*a))*K1p

    I1 = special.iv(index2, kappa*a)
    I1p = index/(kappa*a)*I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa*a)*sqrt(pi/(2*kappa*a))
    i1p = -sqrt(pi/2)*1/(2*(kappa*a)**(3/2.))*special.iv(index, kappa*a) + sqrt(pi/(2*kappa*a))*I1p

    B = zeros((N,N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n>=nu and m>=nu:
                    g1 = gamma(n-nu+0.5)
                    g2 = gamma(m-nu+0.5)
                    g3 = gamma(nu+0.5)
                    g4 = gamma(m+n-nu+1.5)
                    f1 = factorial(n+m-nu)
                    f2 = factorial(n-nu)
                    f3 = factorial(m-nu)
                    f4 = factorial(nu)
                    Anm = g1*g2*g3*f1*(n+m-2*nu+0.5)/(pi*g4*f2*f3*f4)
                    kB = special.kv(n+m-2*nu+0.5,kappa*R)*sqrt(pi/(2*kappa*R))
                    B[n,m] += Anm*kB 

    M = zeros((N,N), float)
    E_hat = E_1/E_2
    for i in range(N):
        for j in range(N):
            M[i,j] = (2*i+1)*B[i,j]*(kappa*i1p[i] - E_hat*i*i1[i]/a)
            if i==j:
                M[i,j] += kappa*k1p[i] - E_hat*i*k1[i]/a

    RHS = zeros(N)
    RHS[0] = -E_hat*q/(4*pi*E_1*a*a)

    a_coeff = solve(M,RHS)

    a0 = a_coeff[0] 
    a0_inf = -E_hat*q/(4*pi*E_1*a*a)*1/(kappa*k1p[0])
   
    phi_2 = a0*k1[0] + i1[0]*sum(a_coeff*B[:,0]) - q/(4*pi*E_1*a)
    phi_1 = a0_inf*k1[0] - q/(4*pi*E_1*a)
    phi_inter = phi_2-phi_1 

    CC0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)

    Einter = 0.5*CC0*q*phi_inter
    E1sphere = 0.5*CC0*q*phi_1
    E2sphere = 0.5*CC0*q*phi_2

    return Einter, E1sphere, E2sphere


def constant_potential_single_point(phi0, a, r, kappa):
    phi = a/r * phi0 * exp(kappa*(a-r))
    return phi

def constant_charge_single_point(sigma0, a, r, kappa, epsilon):
    dphi0 = -sigma0/epsilon
    phi = -dphi0 * a*a/(1+kappa*a) * exp(kappa*(a-r))/r 
    return phi

def constant_potential_single_charge(phi0, radius, kappa, epsilon):
    dphi = -phi0*((1.+kappa*radius)/radius)
    sigma = -epsilon*dphi # Surface charge
    return sigma

def constant_charge_single_potential(sigma0, radius, kappa, epsilon):
    dphi = -sigma0/epsilon 
    phi = -dphi * radius/(1.+kappa*radius) # Surface potential
    return phi

def constant_charge_twosphere_HsuLiu(sigma01, sigma02, r1, r2, R, kappa, epsilon):
    
    gamma1 = -0.5*(1/(kappa*r1) - (1 + 1/(kappa*r1))*exp(-2*kappa*r1))
    gamma2 = -0.5*(1/(kappa*r2) - (1 + 1/(kappa*r2))*exp(-2*kappa*r2))

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    f1 = (0.5+gamma1)/(0.5-gamma1)
    f2 = (0.5+gamma2)/(0.5-gamma2)

    if f1*f2<0:
        A = arctan(sqrt(abs(f1*f2))*exp(-kappa*(R-r1-r2)))
    else:
        A = arctanh(sqrt(f1*f2)*exp(-kappa*(R-r1-r2)))

    phi01 = constant_charge_single_potential(sigma01, r1, kappa, epsilon)
    phi02 = constant_charge_single_potential(sigma02, r2, kappa, epsilon)

    C0 = pi*epsilon*r1*r2/R
    C1 = (f2*phi01*phi01 + f1*phi02*phi02)/(f1*f2) * log(1-f1*f2*exp(-2*kappa*(R-r1-r2)))
    C2 = 4*phi01*phi02/sqrt(abs(f1*f2)) * A

    CC0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)

    E_inter = CC0*C0*(C1 + C2)

    return E_inter

def constant_charge_twosphere_bell(sigma01, sigma02, r1, r2, R, kappa, epsilon):

    E_inter = 4*pi/epsilon*(sigma01*r1*r1/(1+kappa*r1))*(sigma02*r2*r2/(1+kappa*r2))*exp(-kappa*(R-r1-r2))/R

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 
    CC0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)

    return CC0*E_inter

def constant_potential_twosphere(phi01, phi02, r1, r2, R, kappa, epsilon):

    kT = 4.1419464e-21 # at 300K
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 
    C0 = kT/qe

    phi01 /= C0
    phi02 /= C0

    k1 = special.kv(0.5,kappa*r1)*sqrt(pi/(2*kappa*r1))
    k2 = special.kv(0.5,kappa*r2)*sqrt(pi/(2*kappa*r2))
    B00 = special.kv(0.5,kappa*R)*sqrt(pi/(2*kappa*R))
#    k1 = special.kv(0.5,kappa*r1)*sqrt(2/(pi*kappa*r1))
#    k2 = special.kv(0.5,kappa*r2)*sqrt(2/(pi*kappa*r2))
#    B00 = special.kv(0.5,kappa*R)*sqrt(2/(pi*kappa*R))

    i1 = special.iv(0.5,kappa*r1)*sqrt(pi/(2*kappa*r1))
    i2 = special.iv(0.5,kappa*r2)*sqrt(pi/(2*kappa*r2))

    a0 = (phi02*B00*i1 - phi01*k2)/(B00*B00*i2*i1 - k1*k2)
    b0 = (phi02*k1 - phi01*B00*i2)/(k2*k1 - B00*B00*i1*i2)

    U1 = 2*pi*phi01*(phi01*exp(kappa*r1)*(kappa*r1)*(kappa*r1)/sinh(kappa*r1) - pi*a0/(2*i1))
    U2 = 2*pi*phi02*(phi02*exp(kappa*r2)*(kappa*r2)*(kappa*r2)/sinh(kappa*r2) - pi*b0/(2*i2))

    print 'U1: %f'%U1
    print 'U2: %f'%U2
    print 'E: %f'%(U1 + U2) 
    C1 = C0*C0*epsilon/kappa
    u1 = U1*C1
    u2 = U2*C1

    CC0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)

    E_inter = CC0*(u1+u2)

    return E_inter

def constant_potential_twosphere_2(phi01, phi02, r1, r2, R, kappa, epsilon):

    kT = 4.1419464e-21 # at 300K
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 
    h = R-r1-r2
#    E_inter = r1*r2*epsilon/(4*R) * ( (phi01+phi02)**2 * log(1+exp(-kappa*h)) + (phi01-phi02)**2*log(1-exp(-kappa*h)) )
#    E_inter = epsilon*r1*phi01**2/2 * log(1+exp(-kappa*h))
    E_inter = epsilon*r1*r2*(phi01**2+phi02**2)/(4*(r1+r2)) * ( (2*phi01*phi02)/(phi01**2+phi02**2) * log((1+exp(-kappa*h))/(1-exp(-kappa*h))) + log(1-exp(-2*kappa*h)) )

    CC0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E_inter *= CC0
    return E_inter

def constant_potential_single_energy(phi0, r1, kappa, epsilon):

    N = 1 # Number of terms in expansion
     
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    index2 = arange(N+1, dtype=float) + 0.5
    index  = index2[0:-1]

    K1 = special.kv(index2, kappa*r1)
    K1p = index/(kappa*r1)*K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    k1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.kv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*K1p

    a0_inf = phi0/k1[0]
    U1_inf = a0_inf*k1p[0]
 
    C1 = 2*pi*kappa*phi0*r1*r1*epsilon
    C0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E = C0*C1*U1_inf
    
    return E

def constant_charge_single_energy(phi0, r1, kappa, epsilon):

    N = 20 # Number of terms in expansion
     
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    index2 = arange(N+1, dtype=float) + 0.5
    index  = index2[0:-1]

    K1 = special.kv(index2, kappa*r1)
    K1p = index/(kappa*r1)*K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    k1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.kv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*K1p

    a0_inf = -phi0/(epsilon*kappa*k1p[0])
   
    U1_inf = a0_inf*k1[0]
 
    C1 = 2*pi*phi0*r1*r1
    C0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E_inter = C0*C1*U1_inf
    
    return E_inter

def constant_potential_twosphere_dissimilar(phi01, phi02, r1, r2, R, kappa, epsilon):

    N = 20 # Number of terms in expansion
     
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    index2 = arange(N+1, dtype=float) + 0.5
    index  = index2[0:-1]

    K1 = special.kv(index2, kappa*r1)
    K1p = index/(kappa*r1)*K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    k1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.kv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*K1p

    K2 = special.kv(index2, kappa*r2)
    K2p = index/(kappa*r2)*K2[0:-1] - K2[1:]
    k2 = special.kv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))
    k2p = -sqrt(pi/2)*1/(2*(kappa*r2)**(3/2.))*special.kv(index, kappa*r2) + sqrt(pi/(2*kappa*r2))*K2p

    I1 = special.iv(index2, kappa*r1)
    I1p = index/(kappa*r1)*I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    i1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.iv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*I1p

    I2 = special.iv(index2, kappa*r2)
    I2p = index/(kappa*r2)*I2[0:-1] + I2[1:]
    i2 = special.iv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))
    i2p = -sqrt(pi/2)*1/(2*(kappa*r2)**(3/2.))*special.iv(index, kappa*r2) + sqrt(pi/(2*kappa*r2))*I2p

    B = zeros((N,N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n>=nu and m>=nu:
                    g1 = gamma(n-nu+0.5)
                    g2 = gamma(m-nu+0.5)
                    g3 = gamma(nu+0.5)
                    g4 = gamma(m+n-nu+1.5)
                    f1 = factorial(n+m-nu)
                    f2 = factorial(n-nu)
                    f3 = factorial(m-nu)
                    f4 = factorial(nu)
                    Anm = g1*g2*g3*f1*(n+m-2*nu+0.5)/(pi*g4*f2*f3*f4)
                    kB = special.kv(n+m-2*nu+0.5,kappa*R)*sqrt(pi/(2*kappa*R))
                    B[n,m] += Anm*kB 

    M = zeros((2*N,2*N), float)
    for j in range(N):
        for n in range(N):
            M[j,n+N] = (2*j+1)*B[j,n]*i1[j]/k2[n]
            M[j+N,n] = (2*j+1)*B[j,n]*i2[j]/k1[n]
            if n==j:
                M[j,n] = 1
                M[j+N,n+N] = 1

    RHS = zeros(2*N)
    RHS[0] = phi01
    RHS[N] = phi02

    coeff = solve(M,RHS)

    a = coeff[0:N]/k1
    b = coeff[N:2*N]/k2

    a0 = a[0] 
    a0_inf = phi01/k1[0]
    b0 = b[0] 
    b0_inf = phi02/k2[0]
   
    U1_inf = a0_inf*k1p[0]
    U1_h   = a0*k1p[0]+i1p[0]*sum(b*B[:,0])
 
    U2_inf = b0_inf*k2p[0]
    U2_h   = b0*k2p[0]+i2p[0]*sum(a*B[:,0])

    C1 = 2*pi*kappa*phi01*r1*r1*epsilon
    C2 = 2*pi*kappa*phi02*r2*r2*epsilon
    C0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E_inter = C0*(C1*(U1_h-U1_inf) + C2*(U2_h-U2_inf))
    
    return E_inter

def constant_charge_twosphere_dissimilar(phi01, phi02, r1, r2, R, kappa, epsilon):

    N = 20 # Number of terms in expansion
     
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    index2 = arange(N+1, dtype=float) + 0.5
    index  = index2[0:-1]

    K1 = special.kv(index2, kappa*r1)
    K1p = index/(kappa*r1)*K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    k1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.kv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*K1p

    K2 = special.kv(index2, kappa*r2)
    K2p = index/(kappa*r2)*K2[0:-1] - K2[1:]
    k2 = special.kv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))
    k2p = -sqrt(pi/2)*1/(2*(kappa*r2)**(3/2.))*special.kv(index, kappa*r2) + sqrt(pi/(2*kappa*r2))*K2p

    I1 = special.iv(index2, kappa*r1)
    I1p = index/(kappa*r1)*I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    i1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.iv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*I1p

    I2 = special.iv(index2, kappa*r2)
    I2p = index/(kappa*r2)*I2[0:-1] + I2[1:]
    i2 = special.iv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))
    i2p = -sqrt(pi/2)*1/(2*(kappa*r2)**(3/2.))*special.iv(index, kappa*r2) + sqrt(pi/(2*kappa*r2))*I2p

    B = zeros((N,N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n>=nu and m>=nu:
                    g1 = gamma(n-nu+0.5)
                    g2 = gamma(m-nu+0.5)
                    g3 = gamma(nu+0.5)
                    g4 = gamma(m+n-nu+1.5)
                    f1 = factorial(n+m-nu)
                    f2 = factorial(n-nu)
                    f3 = factorial(m-nu)
                    f4 = factorial(nu)
                    Anm = g1*g2*g3*f1*(n+m-2*nu+0.5)/(pi*g4*f2*f3*f4)
                    kB = special.kv(n+m-2*nu+0.5,kappa*R)*sqrt(pi/(2*kappa*R))
                    B[n,m] += Anm*kB 

    M = zeros((2*N,2*N), float)
    for j in range(N):
        for n in range(N):
            M[j,n+N] = (2*j+1)*B[j,n]*r1*i1p[j]/(r2*k2p[n])
            M[j+N,n] = (2*j+1)*B[j,n]*r2*i2p[j]/(r1*k1p[n])
            if n==j:
                M[j,n] = 1
                M[j+N,n+N] = 1

    RHS = zeros(2*N)
    RHS[0] = phi01*r1/epsilon
    RHS[N] = phi02*r2/epsilon

    coeff = solve(M,RHS)

    a = coeff[0:N]/(-r1*kappa*k1p)
    b = coeff[N:2*N]/(-r2*kappa*k2p)

    a0 = a[0] 
    a0_inf = -phi01/(epsilon*kappa*k1p[0])
    b0 = b[0] 
    b0_inf = -phi02/(epsilon*kappa*k2p[0])
   
    U1_inf = a0_inf*k1[0]
    U1_h   = a0*k1[0]+i1[0]*sum(b*B[:,0])
 
    U2_inf = b0_inf*k2[0]
    U2_h   = b0*k2[0]+i2[0]*sum(a*B[:,0])

    C1 = 2*pi*phi01*r1*r1
    C2 = 2*pi*phi02*r2*r2
    C0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E_inter = C0*(C1*(U1_h-U1_inf) + C2*(U2_h-U2_inf))
    
    return E_inter

def molecule_constant_potential(q, phi02, r1, r2, R, kappa, E_1, E_2):

    N = 20 # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    index2 = arange(N+1, dtype=float) + 0.5
    index  = index2[0:-1]

    K1 = special.kv(index2, kappa*r1)
    K1p = index/(kappa*r1)*K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    k1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.kv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*K1p

    K2 = special.kv(index2, kappa*r2)
    K2p = index/(kappa*r2)*K2[0:-1] - K2[1:]
    k2 = special.kv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))
    k2p = -sqrt(pi/2)*1/(2*(kappa*r2)**(3/2.))*special.kv(index, kappa*r2) + sqrt(pi/(2*kappa*r2))*K2p

    I1 = special.iv(index2, kappa*r1)
    I1p = index/(kappa*r1)*I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    i1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.iv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*I1p

    I2 = special.iv(index2, kappa*r2)
    I2p = index/(kappa*r2)*I2[0:-1] + I2[1:]
    i2 = special.iv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))
    i2p = -sqrt(pi/2)*1/(2*(kappa*r2)**(3/2.))*special.iv(index, kappa*r2) + sqrt(pi/(2*kappa*r2))*I2p

    B = zeros((N,N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n>=nu and m>=nu:
                    g1 = gamma(n-nu+0.5)
                    g2 = gamma(m-nu+0.5)
                    g3 = gamma(nu+0.5)
                    g4 = gamma(m+n-nu+1.5)
                    f1 = factorial(n+m-nu)
                    f2 = factorial(n-nu)
                    f3 = factorial(m-nu)
                    f4 = factorial(nu)
                    Anm = g1*g2*g3*f1*(n+m-2*nu+0.5)/(pi*g4*f2*f3*f4)
                    kB = special.kv(n+m-2*nu+0.5,kappa*R)*sqrt(pi/(2*kappa*R))
                    B[n,m] += Anm*kB 

    E_hat = E_1/E_2
    M = zeros((2*N,2*N), float)
    for j in range(N):
        for n in range(N):
            M[j,n+N] = (2*j+1)*B[j,n]*(kappa*i1p[j]/k2[n] - E_hat*j/r1*i1[j]/k2[n])
            M[j+N,n] = (2*j+1)*B[j,n]*i2[j] * 1/(kappa*k1p[n] - E_hat*n/r1*k1[n])
            if n==j:
                M[j,n] = 1
                M[j+N,n+N] = 1

    RHS = zeros(2*N)
    RHS[0] = -E_hat*q/(4*pi*E_1*r1*r1)
    RHS[N] = phi02

    coeff = solve(M,RHS)

    a = coeff[0:N]/(kappa*k1p - E_hat*arange(N)/r1*k1)
    b = coeff[N:2*N]/k2

    a0 = a[0] 
    a0_inf = -E_hat*q/(4*pi*E_1*r1*r1)*1/(kappa*k1p[0]) 
    b0 = b[0] 
    b0_inf = phi02/k2[0]
   
    phi_inf = a0_inf*k1[0] - q/(4*pi*E_1*r1)
    phi_h   = a0*k1[0] + i1[0]*sum(b*B[:,0]) - q/(4*pi*E_1*r1) 
    phi_inter = phi_h - phi_inf
 
    U_inf = b0_inf*k2p[0]
    U_h   = b0*k2p[0]+i2p[0]*sum(a*B[:,0])
    U_inter = U_h - U_inf

    C0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    C1 = q * 0.5
    C2 = 2*pi*kappa*phi02*r2*r2*E_2
    E_inter = C0*(C1*phi_inter + C2*U_inter)
    
    return E_inter

def molecule_constant_charge(q, phi02, r1, r2, R, kappa, E_1, E_2):

    N = 20 # Number of terms in expansion
     
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    index2 = arange(N+1, dtype=float) + 0.5
    index  = index2[0:-1]

    K1 = special.kv(index2, kappa*r1)
    K1p = index/(kappa*r1)*K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    k1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.kv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*K1p

    K2 = special.kv(index2, kappa*r2)
    K2p = index/(kappa*r2)*K2[0:-1] - K2[1:]
    k2 = special.kv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))
    k2p = -sqrt(pi/2)*1/(2*(kappa*r2)**(3/2.))*special.kv(index, kappa*r2) + sqrt(pi/(2*kappa*r2))*K2p

    I1 = special.iv(index2, kappa*r1)
    I1p = index/(kappa*r1)*I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    i1p = -sqrt(pi/2)*1/(2*(kappa*r1)**(3/2.))*special.iv(index, kappa*r1) + sqrt(pi/(2*kappa*r1))*I1p

    I2 = special.iv(index2, kappa*r2)
    I2p = index/(kappa*r2)*I2[0:-1] + I2[1:]
    i2 = special.iv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))
    i2p = -sqrt(pi/2)*1/(2*(kappa*r2)**(3/2.))*special.iv(index, kappa*r2) + sqrt(pi/(2*kappa*r2))*I2p

    B = zeros((N,N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n>=nu and m>=nu:
                    g1 = gamma(n-nu+0.5)
                    g2 = gamma(m-nu+0.5)
                    g3 = gamma(nu+0.5)
                    g4 = gamma(m+n-nu+1.5)
                    f1 = factorial(n+m-nu)
                    f2 = factorial(n-nu)
                    f3 = factorial(m-nu)
                    f4 = factorial(nu)
                    Anm = g1*g2*g3*f1*(n+m-2*nu+0.5)/(pi*g4*f2*f3*f4)
                    kB = special.kv(n+m-2*nu+0.5,kappa*R)*sqrt(pi/(2*kappa*R))
                    B[n,m] += Anm*kB 

    E_hat = E_1/E_2
    M = zeros((2*N,2*N), float)
    for j in range(N):
        for n in range(N):
            M[j,n+N] = (2*j+1)*B[j,n]*(i1p[j]/k2p[n] - E_hat*j/r1*i1[j]/(kappa*k2p[n]))
            M[j+N,n] = (2*j+1)*B[j,n]*i2p[j]*kappa * 1/(kappa*k1p[n] - E_hat*n/r1*k1[n])
            if n==j:
                M[j,n] = 1
                M[j+N,n+N] = 1

    RHS = zeros(2*N)
    RHS[0] = -E_hat*q/(4*pi*E_1*r1*r1)
    RHS[N] = -phi02/E_2

    coeff = solve(M,RHS)

    a = coeff[0:N]/(kappa*k1p - E_hat*arange(N)/r1*k1)
    b = coeff[N:2*N]/(kappa*k2p)

    a0 = a[0] 
    a0_inf = -E_hat*q/(4*pi*E_1*r1*r1)*1/(kappa*k1p[0]) 
    b0 = b[0] 
    b0_inf = -phi02/(E_2*kappa*k2p[0])
   
    phi_inf = a0_inf*k1[0] - q/(4*pi*E_1*r1)
    phi_h   = a0*k1[0] + i1[0]*sum(b*B[:,0]) - q/(4*pi*E_1*r1) 
    phi_inter = phi_h - phi_inf
 
    U_inf = b0_inf*k2[0]
    U_h   = b0*k2[0]+i2[0]*sum(a*B[:,0])
    U_inter = U_h - U_inf

    C0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    C1 = q * 0.5
    C2 = 2*pi*phi02*r2*r2
    E_inter = C0*(C1*phi_inter + C2*U_inter)
    
    return E_inter


def constant_potential_twosphere_identical(phi01, phi02, r1, r2, R, kappa, epsilon):
#   From Carnie+Chan 1993

    N = 20 # Number of terms in expansion
    
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    index = arange(N, dtype=float) + 0.5

    k1 = special.kv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    k2 = special.kv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))

    i1 = special.iv(index, kappa*r1)*sqrt(pi/(2*kappa*r1))
    i2 = special.iv(index, kappa*r2)*sqrt(pi/(2*kappa*r2))

    B = zeros((N,N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n>=nu and m>=nu:
                    g1 = gamma(n-nu+0.5)
                    g2 = gamma(m-nu+0.5)
                    g3 = gamma(nu+0.5)
                    g4 = gamma(m+n-nu+1.5)
                    f1 = factorial(n+m-nu)
                    f2 = factorial(n-nu)
                    f3 = factorial(m-nu)
                    f4 = factorial(nu)
                    Anm = g1*g2*g3*f1*(n+m-2*nu+0.5)/(pi*g4*f2*f3*f4)
                    kB = special.kv(n+m-2*nu+0.5,kappa*R)*sqrt(pi/(2*kappa*R))
                    B[n,m] += Anm*kB 

    M = zeros((N,N), float)
    for i in range(N):
        for j in range(N):
            M[i,j] = (2*i+1)*B[i,j]*i1[i]
            if i==j:
                M[i,j] += k1[i]

    RHS = zeros(N)
    RHS[0] = phi01

    a = solve(M,RHS)

    a0 = a[0] 
   
    U = 4*pi * ( -pi/2 * a0/phi01 * 1/sinh(kappa*r1) + kappa*r1 + kappa*r1/tanh(kappa*r1) )

#    print 'E: %f'%U
    C0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    C1 = r1*epsilon*phi01*phi01
    E_inter = U*C1*C0
                            
    return E_inter

def constant_charge_twosphere_identical(sigma, a, R, kappa, epsilon):
#   From Carnie+Chan 1993

    N = 10 # Number of terms in expansion
    E_p = 0 # Permitivitty inside sphere
    
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    index2 = arange(N+1, dtype=float) + 0.5
    index  = index2[0:-1]

    K1 = special.kv(index2, kappa*a)
    K1p = index/(kappa*a)*K1[0:-1] - K1[1:]
    
    k1 = special.kv(index, kappa*a)*sqrt(pi/(2*kappa*a))
    k1p = -sqrt(pi/2)*1/(2*(kappa*a)**(3/2.))*special.kv(index, kappa*a) + sqrt(pi/(2*kappa*a))*K1p

    I1 = special.iv(index2, kappa*a)
    I1p = index/(kappa*a)*I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa*a)*sqrt(pi/(2*kappa*a))
    i1p = -sqrt(pi/2)*1/(2*(kappa*a)**(3/2.))*special.iv(index, kappa*a) + sqrt(pi/(2*kappa*a))*I1p


    B = zeros((N,N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n>=nu and m>=nu:
                    g1 = gamma(n-nu+0.5)
                    g2 = gamma(m-nu+0.5)
                    g3 = gamma(nu+0.5)
                    g4 = gamma(m+n-nu+1.5)
                    f1 = factorial(n+m-nu)
                    f2 = factorial(n-nu)
                    f3 = factorial(m-nu)
                    f4 = factorial(nu)
                    Anm = g1*g2*g3*f1*(n+m-2*nu+0.5)/(pi*g4*f2*f3*f4)
                    kB = special.kv(n+m-2*nu+0.5,kappa*R)*sqrt(pi/(2*kappa*R))
                    B[n,m] += Anm*kB 

    M = zeros((N,N), float)
    for i in range(N):
        for j in range(N):
            M[i,j] = (2*i+1)*B[i,j]*(E_p/epsilon*i*i1[i] - a*kappa*i1p[i])
            if i==j:
                M[i,j] += (E_p/epsilon*i*k1[i] - a*kappa*k1p[i])

    RHS = zeros(N)
    RHS[0] = a*sigma/epsilon

    a_coeff = solve(M,RHS)

    a0 = a_coeff[0] 
   
    C0 = a*sigma/epsilon
    CC0 = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    
    E_inter = 4*pi*a*epsilon*C0*C0*CC0( pi*a0/(2*C0*(kappa*a*cosh(kappa*a)-sinh(kappa*a))) - 1/(1+kappa*a) - 1/(kappa*a*1/tanh(kappa*a)-1) )

    return E_inter



'''
r1 = 1.
phi01 = 1.
r2 = 2.
phi02 = 2.

R = 5

kappa = 0.1
epsilon = 80.

E_inter = constant_potential_twosphere(phi01, phi02, r1, r2, R, kappa, epsilon)
print E_inter
'''



q   = array([[1.],[-1.],[-1.]])
p   = array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
#p   = array([[0.,1.,0.]])
Q   = array([[[1.,0.,0.],[0.,-1.,0.],[0.,0.,0.]],[[0.,0.,0.],[0.,1.,0.],[0.,0.,-1.]],[[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]]])
#Q   = array([[[1.,0.,0.],[0.,-1.,0.],[0.,0.,0.]]])
alpha = array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])*10
#alpha = array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])*0.
xq  = array([[1e-12,1e-12,1e-12],[1.,1.41421356,1.],[-1.,-1.,1.41421356]])
#xq  = array([[1.,1.,1.41421356]])
#xq  = array([[1e-12,1e-12,1e-12]])
E_1 = 4.
E_2 = 80.
E_0 = 8.854187818e-12
R   = 4.
N   = 15
Na  = 6.0221415e23
a   = R
kappa = 0.125

#energy_sph = an_spherical(q, xq, E_1, E_2, R, N)
#energy = an_P(q, xq, E_1, E_2, R, kappa, a, N)
#energy_mult = an_multipole(q, p, Q, xq, E_1, E_2, R, N)
#energy_mult_pol, Epol, p_pol = an_multipole_polarizable(q, p, Q, alpha, xq, E_1, E_2, kappa, R, a, N)
#energy_mult2 = an_multipole_2(q, p, Q, xq, E_1, E_2, R, N)

energy_mult_pol = solvation_energy_polarizable(q, p, Q, alpha, xq, E_1, E_2, kappa, R, a, N)

#Ecoul =  coulomb_energy(q, p, Q, xq, E_1)
#print Ecoul
#print energy
#print energy_sph
#print energy_mult
print energy_mult_pol
#print energy_mult2

#JtoCal = 4.184    
#E_solv_sph = 0.5*sum(q*PHI_sph)*Na*1e7/JtoCal
#E_solv_P = 0.5*sum(q*PHI_P)*Na*1e7/JtoCal
#print 'With spherical harmonics: %f'%E_solv_sph
#print 'With Legendre functions : %f'%E_solv_P

