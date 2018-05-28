/* 
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
*/ 

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
#define REAL double

double get_time (void)
{
    struct timeval tv; 
    gettimeofday(&tv,NULL);
    return (double)(tv.tv_sec+1e-6*tv.tv_usec);
}

REAL norm(REAL *x)
{
    return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}

void cross(REAL *x, REAL *y, REAL *z) // z is the resulting array
{
    z[0] = x[1]*y[2] - x[2]*y[1];
    z[1] = x[2]*y[0] - x[0]*y[2];
    z[2] = x[0]*y[1] - x[1]*y[0];
}

void MV(REAL *M, REAL *V, REAL *res) // 3x3 mat-vec
{
    REAL V2[3] = {V[0], V[1], V[2]};
    for (int i=0; i<3; i++)
    {
        REAL sum = 0.;
        for (int j=0; j<3; j++)
        {
            sum += M[3*i+j]*V2[j]; 
        }
        res[i] = sum;
    }
}

REAL dot_prod(REAL *x, REAL *y) // len(3) vector dot product
{
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}

void axpy(REAL *x, REAL *y, REAL *z, REAL alpha, int sign, int N)
{
    for(int i=0; i<N; i++)
    {
        z[i] = sign*alpha*x[i] + y[i];
    }
}

void ax(REAL *x, REAL *y, REAL alpha, int N)
{
    for(int i=0; i<N; i++)
    {
        y[i] = alpha*x[i];
    }

}

void lineInt(REAL &PHI_K, REAL &PHI_V, REAL z, REAL x, REAL v1, REAL v2, REAL kappa, REAL *xk, REAL *wk, int K, int LorY)
{
    REAL theta1 = atan2(v1,x);
    REAL theta2 = atan2(v2,x);
    REAL dtheta = theta2 - theta1;
    REAL thetam = (theta2 + theta1)/2; 


    REAL absZ = fabs(z), signZ;
    if (absZ<1e-10) signZ = 0;
    else            signZ = z/absZ;

    // Loop over gauss points
    REAL thetak, Rtheta, R, expKr, expKz;
    if (LorY==2)
        expKz = exp(-kappa*absZ);

    for (int i=0; i<K; i++)
    {
        thetak = dtheta/2*xk[i] + thetam;
        Rtheta = x/cos(thetak);
        R      = sqrt(Rtheta*Rtheta + z*z);
        expKr  = exp(-kappa*R);
        if (LorY==2)
        {
            if (kappa>1e-12)
            {
                PHI_V += -wk[i]*(expKr - expKz)/kappa * dtheta/2;
                PHI_K +=  wk[i]*(z/R*expKr - expKz*signZ) * dtheta/2;
            }
            else
            {
                PHI_V +=  wk[i]*(R-absZ) * dtheta/2;
                PHI_K +=  wk[i]*(z/R - signZ) * dtheta/2;
            }
        }
        if (LorY==1)
        {
            PHI_V +=  wk[i]*(R-absZ) * dtheta/2;
            PHI_K +=  wk[i]*(z/R - signZ) * dtheta/2;
        }
    }
}

void intSide(REAL &PHI_K, REAL &PHI_V, REAL *v1, REAL *v2, REAL p, REAL kappa, REAL *xk, REAL *wk, int K, int LorY)
{
    REAL v21[3];
    for (int i=0; i<3; i++)
    {
        v21[i] = v2[i] - v1[i];
    }

    REAL L21 = norm(v21);
    REAL v21u[3];
    ax(v21, v21u, 1/L21, 3);

    REAL unit[3] = {0.,0.,1.};
    REAL orthog[3];
    cross(unit, v21u, orthog);

    REAL alpha = dot_prod(v21,v1)/(L21*L21);

    REAL rOrthog[3];
    axpy(v21, v1, rOrthog, alpha, -1, 3);

    REAL d_toEdge = norm(rOrthog);
    REAL v1_neg[3];
    ax(v1, v1_neg, -1, 3);
    
    REAL side_vec[3];
    cross(v21, v1_neg, side_vec);

    REAL rotateToVertLine[9];

    for(int i=0; i<3; i++)
    {
        rotateToVertLine[3*i] = orthog[i];
        rotateToVertLine[3*i+1] = v21u[i];
        rotateToVertLine[3*i+2] = unit[i];
    }

    REAL v1new[3];
    MV(rotateToVertLine,v1,v1new);

    if (v1new[0]<0)
    {
        ax(v21u, v21u, -1, 3);
        ax(orthog, orthog, -1, 3);
        ax(rotateToVertLine, rotateToVertLine, -1, 9);
        rotateToVertLine[8] = 1.;
        MV(rotateToVertLine,v1,v1new);
    }

    REAL v2new[3], rOrthognew[3];
    MV(rotateToVertLine,v2,v2new);
    MV(rotateToVertLine,rOrthog,rOrthognew);
    REAL x = v1new[0];

    if ((v1new[1]>0 && v2new[1]<0) || (v1new[1]<0 && v2new[1]>0))
    {
        REAL PHI1_K = 0. , PHI2_K = 0.;
        REAL PHI1_V = 0. , PHI2_V = 0.;
        lineInt(PHI1_K, PHI1_V, p, x, 0, v1new[1], kappa, xk, wk, K, LorY);
        lineInt(PHI2_K, PHI2_V, p, x, v2new[1], 0, kappa, xk, wk, K, LorY);

        PHI_K += PHI1_K + PHI2_K;
        PHI_V += PHI1_V + PHI2_V;
    }   
    else
    {
        REAL PHI_Kaux = 0., PHI_Vaux = 0.;
        lineInt(PHI_Kaux, PHI_Vaux, p, x, v1new[1], v2new[1], kappa, xk, wk, K, LorY);

        PHI_K -= PHI_Kaux;
        PHI_V -= PHI_Vaux;
    }

}


void SA(REAL &PHI_K, REAL &PHI_V, REAL *y, REAL *x, REAL kappa, int same, 
        REAL K_diag, REAL V_diag, int LorY, REAL *xk, int xkSize, REAL *wk)
{
    // Put first panel at origin
    REAL y0_panel[3], y1_panel[3], y2_panel[3], x_panel[3];
    REAL X[3], Y[3], Z[3];
    for (int i=0; i<3;i++)
    {
        x_panel[i] = x[i] - y[i];
        y0_panel[i] = 0.; 
        y1_panel[i] = y[3+i] - y[i]; 
        y2_panel[i] = y[6+i] - y[i]; 
        X[i] = y1_panel[i]; 
    }

    // Find panel coordinate system X: 0->1
    cross(y1_panel, y2_panel, Z);
    REAL Xnorm = norm(X); 
    REAL Znorm = norm(Z); 
    for (int i=0; i<3; i++)
    {
        X[i] /= Xnorm;
        Z[i] /= Znorm;
    }

    cross(Z,X,Y);

    // Rotate the coordinate system to match panel plane
    REAL rot_matrix[9];
    for (int i=0; i<3; i++)
    {
        rot_matrix[i] = X[i];
        rot_matrix[i+3] = Y[i];
        rot_matrix[i+6] = Z[i];
    }
    
    REAL panel0_plane[3], panel1_plane[3], panel2_plane[3], x_plane[3];
    MV(rot_matrix, y0_panel, panel0_plane);
    MV(rot_matrix, y1_panel, panel1_plane);
    MV(rot_matrix, y2_panel, panel2_plane);
    MV(rot_matrix, x_panel, x_plane);

    // Shift origin so it matches collocation point
    REAL panel0_final[3], panel1_final[3], panel2_final[3];
    for (int i=0; i<3; i++)
    {
        if (i<2)
        {
            panel0_final[i] = panel0_plane[i] - x_plane[i]; 
            panel1_final[i] = panel1_plane[i] - x_plane[i]; 
            panel2_final[i] = panel2_plane[i] - x_plane[i]; 
        }
        else
        {
            panel0_final[i] = panel0_plane[i]; 
            panel1_final[i] = panel1_plane[i]; 
            panel2_final[i] = panel2_plane[i]; 
        }
    }

    // Loop over sides
    intSide(PHI_K, PHI_V, panel0_final, panel1_final, x_plane[2], kappa, xk, wk, xkSize, LorY); // Side 0
    intSide(PHI_K, PHI_V, panel1_final, panel2_final, x_plane[2], kappa, xk, wk, xkSize, LorY); // Side 1
    intSide(PHI_K, PHI_V, panel2_final, panel0_final, x_plane[2], kappa, xk, wk, xkSize, LorY); // Side 2

    if (same==1)
    {
        PHI_K += K_diag;
        PHI_V += V_diag;
    }

}

void computeDiagonal(REAL *VL, int VLSize, REAL *KL, int KLSize, REAL *VY, int VYSize, REAL *KY, int KYSize, 
                    REAL *triangle, int triangleSize, REAL *centers, int centersSize, REAL kappa,
                    REAL K_diag, REAL V_diag, REAL *xk, int xkSize, REAL *wk, int wkSize)
{
    int N = VLSize, LorY;
    REAL PHI_K, PHI_V;
    for(int i=0; i<N; i++)
    {
        REAL panel[9] = {triangle[9*i], triangle[9*i+1], triangle[9*i+2],
                 triangle[9*i+3], triangle[9*i+4], triangle[9*i+5],
                 triangle[9*i+6], triangle[9*i+7], triangle[9*i+8]};
        REAL center[3] = {centers[3*i], centers[3*i+1], centers[3*i+2]};
    
        PHI_K = 0.;
        PHI_V = 0.;
        LorY = 1; // Laplace
        SA(PHI_K, PHI_V, panel, center, 1e-12, 1, 
            K_diag, V_diag, LorY, xk, xkSize, wk);

        VL[i] = PHI_V;
        KL[i] = PHI_K;

        PHI_K = 0.;
        PHI_V = 0.;
        LorY = 2; // Yukawa 
        SA(PHI_K, PHI_V, panel, center, kappa, 1, 
            K_diag, V_diag, LorY, xk, xkSize, wk);
        
        VY[i] = PHI_V;
        KY[i] = PHI_K;

    }
}

void GQ_fine(REAL &PHI_K, REAL &PHI_V, REAL *panel, REAL xi, REAL yi, REAL zi, 
            REAL kappa, REAL *Xk, REAL *Wk, int K_fine, REAL Area, int LorY)
{
    REAL nx, ny, nz;
    REAL dx, dy, dz, r, aux;

    PHI_K = 0.;
    PHI_V = 0.;

    aux = 1/(2*Area);
    nx = ((panel[4]-panel[1])*(panel[2]-panel[8]) - (panel[5]-panel[2])*(panel[1]-panel[7])) * aux;
    ny = ((panel[5]-panel[2])*(panel[0]-panel[6]) - (panel[3]-panel[0])*(panel[2]-panel[8])) * aux;
    nz = ((panel[3]-panel[0])*(panel[1]-panel[7]) - (panel[4]-panel[1])*(panel[0]-panel[6])) * aux;


    #pragma unroll
    for (int kk=0; kk<K_fine; kk++)
    {
        dx = xi - (panel[0]*Xk[3*kk] + panel[3]*Xk[3*kk+1] + panel[6]*Xk[3*kk+2]);
        dy = yi - (panel[1]*Xk[3*kk] + panel[4]*Xk[3*kk+1] + panel[7]*Xk[3*kk+2]);
        dz = zi - (panel[2]*Xk[3*kk] + panel[5]*Xk[3*kk+1] + panel[8]*Xk[3*kk+2]);
        r   = 1/sqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!

        if (LorY==1)
        {
            aux = Wk[kk]*Area*r;
            PHI_V += aux;
            PHI_K += aux*(nx*dx+ny*dy+nz*dz)*(r*r);
        }

        else
        {
            aux = Wk[kk]*Area*exp(-kappa*1/r)*r;
            PHI_V += aux;
            PHI_K += aux*(nx*dx+ny*dy+nz*dz)*r*(kappa+r);
        }

    }
}

void GQ_fine_derivative(REAL &dPHI_Kx, REAL &dPHI_Ky, REAL &dPHI_Kz, 
                        REAL &dPHI_Vx, REAL &dPHI_Vy, REAL &dPHI_Vz, 
                        REAL *panel, REAL xi, REAL yi, REAL zi, REAL kappa, 
                        REAL *Xk, REAL *Wk, int K_fine, REAL Area, int LorY)
{
    REAL nx, ny, nz;
    REAL dx, dy, dz, r, r3, aux;

    dPHI_Kx = 0.;
    dPHI_Ky = 0.;
    dPHI_Kz = 0.;
    dPHI_Vx = 0.;
    dPHI_Vy = 0.;
    dPHI_Vz = 0.;

    aux = 1/(2*Area);
    nx = ((panel[4]-panel[1])*(panel[2]-panel[8]) - (panel[5]-panel[2])*(panel[1]-panel[7])) * aux;
    ny = ((panel[5]-panel[2])*(panel[0]-panel[6]) - (panel[3]-panel[0])*(panel[2]-panel[8])) * aux;
    nz = ((panel[3]-panel[0])*(panel[1]-panel[7]) - (panel[4]-panel[1])*(panel[0]-panel[6])) * aux;


    #pragma unroll
    for (int kk=0; kk<K_fine; kk++)
    {
        dx = xi - (panel[0]*Xk[3*kk] + panel[3]*Xk[3*kk+1] + panel[6]*Xk[3*kk+2]);
        dy = yi - (panel[1]*Xk[3*kk] + panel[4]*Xk[3*kk+1] + panel[7]*Xk[3*kk+2]);
        dz = zi - (panel[2]*Xk[3*kk] + panel[5]*Xk[3*kk+1] + panel[8]*Xk[3*kk+2]);
        r  = 1/sqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!
        r3 = r*r*r; 

        if (LorY==1)
        {
            aux = Wk[kk]*Area*r3;
            dPHI_Vx -= dx*aux;
            dPHI_Vy -= dy*aux;
            dPHI_Vz -= dz*aux;
            dPHI_Kx += aux*nx-3*aux*dx*(nx*dx+ny*dy+nz*dz)*(r*r);
            dPHI_Ky += aux*ny-3*aux*dy*(nx*dx+ny*dy+nz*dz)*(r*r);
            dPHI_Kz += aux*nz-3*aux*dz*(nx*dx+ny*dy+nz*dz)*(r*r);
        }

        else    // this else will never fire as this function is only used to calculate energy (always Laplace)
        {
            aux = Wk[kk]*Area*exp(-kappa*1/r)*r;
            dPHI_Vx += aux;
            dPHI_Kx += aux*(nx*dx+ny*dy+nz*dz)*r*(kappa+r);
        }

    }
}

void GQ_fine_2derivative(REAL &dPHI_Kxx, REAL &dPHI_Kxy, REAL &dPHI_Kxz, REAL &dPHI_Kyx, REAL &dPHI_Kyy, REAL &dPHI_Kyz,REAL &dPHI_Kzx, REAL &dPHI_Kzy, REAL &dPHI_Kzz,
                         REAL &dPHI_Vxx, REAL &dPHI_Vxy, REAL &dPHI_Vxz, REAL &dPHI_Vyx, REAL &dPHI_Vyy, REAL &dPHI_Vyz,REAL &dPHI_Vzx, REAL &dPHI_Vzy, REAL &dPHI_Vzz,
                        REAL *panel, REAL xi, REAL yi, REAL zi, REAL kappa, 
                        REAL *Xk, REAL *Wk, int K_fine, REAL Area, int LorY)
{
    REAL nx, ny, nz;
    REAL dx, dy, dz, r, r2, r3, aux;

    dPHI_Kxx = 0.;
    dPHI_Kxy = 0.;
    dPHI_Kxz = 0.;
    dPHI_Kyx = 0.;
    dPHI_Kyy = 0.;
    dPHI_Kyz = 0.;
    dPHI_Kzx = 0.;
    dPHI_Kzy = 0.;
    dPHI_Kzz = 0.;
    dPHI_Vxx = 0.;
    dPHI_Vxy = 0.;
    dPHI_Vxz = 0.;
    dPHI_Vyx = 0.;
    dPHI_Vyy = 0.;
    dPHI_Vyz = 0.;
    dPHI_Vzx = 0.;
    dPHI_Vzy = 0.;
    dPHI_Vzz = 0.;

    aux = 1/(2*Area);
    nx = ((panel[4]-panel[1])*(panel[2]-panel[8]) - (panel[5]-panel[2])*(panel[1]-panel[7])) * aux;
    ny = ((panel[5]-panel[2])*(panel[0]-panel[6]) - (panel[3]-panel[0])*(panel[2]-panel[8])) * aux;
    nz = ((panel[3]-panel[0])*(panel[1]-panel[7]) - (panel[4]-panel[1])*(panel[0]-panel[6])) * aux;


    #pragma unroll
    for (int kk=0; kk<K_fine; kk++)
    {
        dx = xi - (panel[0]*Xk[3*kk] + panel[3]*Xk[3*kk+1] + panel[6]*Xk[3*kk+2]);
        dy = yi - (panel[1]*Xk[3*kk] + panel[4]*Xk[3*kk+1] + panel[7]*Xk[3*kk+2]);
        dz = zi - (panel[2]*Xk[3*kk] + panel[5]*Xk[3*kk+1] + panel[8]*Xk[3*kk+2]);
        r  = 1/sqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!
        r2 = r*r;
        r3 = r2*r; 

        if (LorY==1)
        {
            aux = Wk[kk]*Area*r3;
            dPHI_Vxx += aux*(-1+3*dx*dx*r2);
            dPHI_Vxy += aux*3*dx*dy*r2;
            dPHI_Vxz += aux*3*dx*dz*r2;
            dPHI_Vyx += aux*3*dy*dx*r2;
            dPHI_Vyy += aux*(-1+3*dy*dy*r2);
            dPHI_Vxz += aux*3*dy*dz*r2;
            dPHI_Vzx += aux*3*dz*dx*r2;
            dPHI_Vzy += aux*3*dz*dy*r2;
            dPHI_Vzz += aux*(-1+3*dz*dz*r2);
            dPHI_Kxx += -3*aux*r2*(3*dx*nx + dy*ny + dz*nz - 5*r2*dx*dx*(dx*nx+dy*ny+dz*nz)); 
            dPHI_Kxy += -3*aux*r2*(dx*ny + dy*nx - 5*r2*dx*dy*(dx*nx+dy*ny+dz*nz)); 
            dPHI_Kxz += -3*aux*r2*(dx*nz + dz*nx - 5*r2*dx*dz*(dx*nx+dy*ny+dz*nz)); 
            dPHI_Kyx += -3*aux*r2*(dy*nx + dx*ny - 5*r2*dy*dx*(dx*nx+dy*ny+dz*nz)); 
            dPHI_Kyy += -3*aux*r2*(3*dy*ny + dx*nx + dz*nz - 5*r2*dy*dy*(dx*nx+dy*ny+dz*nz)); 
            dPHI_Kyz += -3*aux*r2*(dy*nz + dz*ny - 5*r2*dy*dz*(dx*nx+dy*ny+dz*nz)); 
            dPHI_Kzx += -3*aux*r2*(dz*nx + dx*nz - 5*r2*dz*dx*(dx*nx+dy*ny+dz*nz)); 
            dPHI_Kzy += -3*aux*r2*(dz*ny + dy*nz - 5*r2*dz*dy*(dx*nx+dy*ny+dz*nz)); 
            dPHI_Kzz += -3*aux*r2*(3*dz*nz + dy*ny + dx*nx - 5*r2*dz*dz*(dx*nx+dy*ny+dz*nz)); 
        }

        else    // this else will never fire as this function is only used to calculate energy (always Laplace)
        {
            aux = Wk[kk]*Area*exp(-kappa*1/r)*r;
            dPHI_Vxx += aux;
            dPHI_Kxx += aux*(nx*dx+ny*dy+nz*dz)*r*(kappa+r);
        }

    }
}

void GQ_fineKt(REAL &PHI_Ktx, REAL &PHI_Kty, REAL &PHI_Ktz, REAL *panel, 
            REAL xi, REAL yi, REAL zi, REAL kappa, REAL *Xk, REAL *Wk, 
            int K_fine, REAL Area, int LorY)
{
    REAL dx, dy, dz, r, aux;

    PHI_Ktx = 0.;
    PHI_Kty = 0.;
    PHI_Ktz = 0.;

    #pragma unroll
    for (int kk=0; kk<K_fine; kk++)
    {
        dx = xi - (panel[0]*Xk[3*kk] + panel[3]*Xk[3*kk+1] + panel[6]*Xk[3*kk+2]);
        dy = yi - (panel[1]*Xk[3*kk] + panel[4]*Xk[3*kk+1] + panel[7]*Xk[3*kk+2]);
        dz = zi - (panel[2]*Xk[3*kk] + panel[5]*Xk[3*kk+1] + panel[8]*Xk[3*kk+2]);
        r   = 1/sqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!

        if (LorY==1)
        {
            aux = Wk[kk]*Area*r*r*r;
            PHI_Ktx -= aux*dx;
            PHI_Kty -= aux*dy;
            PHI_Ktz -= aux*dz;
        }

        else
        {
            aux = Wk[kk]*Area*exp(-kappa*1/r)*r*r*(kappa+r);
            PHI_Ktx -= aux*dx;
            PHI_Kty -= aux*dy;
            PHI_Ktz -= aux*dz;
        }
    }
}

void direct_c(REAL *K_aux, int K_auxSize, REAL *V_aux, int V_auxSize, int LorY, REAL K_diag, REAL V_diag, int IorE, REAL *triangle, int triangleSize,
        int *tri, int triSize, int *k, int kSize, REAL *xi, int xiSize, REAL *yi, int yiSize, 
        REAL *zi, int ziSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, 
        REAL *s_zj, int s_zjSize, REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize, REAL *mKclean, int mKcleanSize, REAL *mVclean, int mVcleanSize,
        int *target, int targetSize,REAL *Area, int AreaSize, REAL *sglInt_int, int sglInt_intSize, REAL *sglInt_ext, int sglInt_extSize, 
        REAL *xk, int xkSize, REAL *wk, int wkSize, REAL *Xsk, int XskSize, REAL *Wsk, int WskSize, 
        REAL kappa, REAL threshold, REAL eps, REAL w0, REAL *aux, int auxSize)
{
    double start,stop;
    int N_target = targetSize;
    int N_source = s_xjSize;
    REAL dx, dy, dz, dx_tri, dy_tri, dz_tri, R, R2, R3, R_tri, expKr;
    bool L_d, same, condition_an, condition_gq;

    for(int i_aux=0; i_aux<N_target; i_aux++)
    {  
        int i = target[i_aux];
        for(int j=0; j<N_source; j++)
        {   
            // Check if panels are far enough for Gauss quadrature
            dx_tri = xt[i_aux] - xi[tri[j]];
            dy_tri = yt[i_aux] - yi[tri[j]];
            dz_tri = zt[i_aux] - zi[tri[j]];
            R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri);
            
            L_d  = (sqrt(2*Area[tri[j]])/(R_tri+eps)>=threshold);
            same = (i==tri[j]);
            condition_an = ((same || L_d) && (k[j]==0));
            condition_gq = (!L_d);

            if(condition_gq)
            {
                //start = get_time();
                dx = xt[i_aux] - s_xj[j];
                dy = yt[i_aux] - s_yj[j];
                dz = zt[i_aux] - s_zj[j];
                R  = sqrt(dx*dx + dy*dy + dz*dz + eps*eps);
                R2 = R*R;
                R3 = R2*R;
                if (LorY==2)
                {
                    expKr = exp(-kappa*R);
                    V_aux[i_aux] += m[j]*expKr/R;
                    K_aux[i_aux] += expKr/R2*(kappa+1/R) * (dx*mx[j] + dy*my[j] + dz*mz[j]);
                }
                if (LorY==1)
                {
                    V_aux[i_aux] += m[j]/R;
                    K_aux[i_aux] += 1/R3*(dx*mx[j] + dy*my[j] + dz*mz[j]);
                }
                //stop = get_time();
                //aux[1] += stop - start;
            }
            
            if(condition_an)
            {
                aux[0] += 1;
                REAL center[3] = {xt[i_aux], yt[i_aux], zt[i_aux]};
                REAL panel[9]  = {triangle[9*tri[j]], triangle[9*tri[j]+1], triangle[9*tri[j]+2],
                                triangle[9*tri[j]+3], triangle[9*tri[j]+4], triangle[9*tri[j]+5],
                                triangle[9*tri[j]+6], triangle[9*tri[j]+7], triangle[9*tri[j]+8]};
                REAL PHI_K = 0., PHI_V = 0.;
                
                start = get_time();

                if (same==1)
                {
                    PHI_K = K_diag;
                    if (IorE==1)
                        PHI_V = sglInt_int[j];
                    else
                        PHI_V = sglInt_ext[j];
                }
                else
                {
                    GQ_fine(PHI_K, PHI_V, panel, xt[i_aux], yt[i_aux], zt[i_aux], kappa, Xsk, Wsk, WskSize, Area[tri[j]], LorY); 
                }


                stop = get_time();
                aux[1] += stop - start;

//                printf("%f \t %f\n",PHI_V,mVclean[j]);

                V_aux[i_aux]  += PHI_V * mVclean[j];
                K_aux[i_aux]  += PHI_K * mKclean[j]; 

            }
        }
    }

}

void direct_c_derivative(REAL *dKx_aux, int dKx_auxSize, REAL *dKy_aux, int dKy_auxSize, REAL *dKz_aux, int dKz_auxSize,
                        REAL *dVx_aux, int dVx_auxSize, REAL *dVy_aux, int dVy_auxSize, REAL *dVz_aux, int dVz_auxSize,
                        int LorY, REAL K_diag, REAL V_diag, int IorE, REAL *triangle, int triangleSize,
                        int *tri, int triSize, int *k, int kSize, REAL *xi, int xiSize, REAL *yi, int yiSize, 
                        REAL *zi, int ziSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, 
                        REAL *s_zj, int s_zjSize, REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
                        REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize, REAL *mKclean, int mKcleanSize, REAL *mVclean, int mVcleanSize,
                        int *target, int targetSize,REAL *Area, int AreaSize, REAL *sglInt_int, int sglInt_intSize, REAL *sglInt_ext, int sglInt_extSize, 
                        REAL *xk, int xkSize, REAL *wk, int wkSize, REAL *Xsk, int XskSize, REAL *Wsk, int WskSize, 
                        REAL kappa, REAL threshold, REAL eps, REAL w0, REAL *aux, int auxSize)
{
    double start,stop;
    int N_target = targetSize;
    int N_source = s_xjSize;
    REAL dx, dy, dz, dx_tri, dy_tri, dz_tri, R, R2, R3, R_tri, expKr;
    bool L_d, same, condition_an, condition_gq;

    for(int i_aux=0; i_aux<N_target; i_aux++)
    {  
        int i = target[i_aux];
        for(int j=0; j<N_source; j++)
        {   
            // Check if panels are far enough for Gauss quadrature
            dx_tri = xt[i_aux] - xi[tri[j]];
            dy_tri = yt[i_aux] - yi[tri[j]];
            dz_tri = zt[i_aux] - zi[tri[j]];
            R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri);
            
            L_d  = (sqrt(2*Area[tri[j]])/(R_tri+eps)>=threshold);
            same = (i==tri[j]);
            condition_an = ((same || L_d) && (k[j]==0));
            condition_gq = (!L_d);

            if(condition_gq)
            {
                //start = get_time();
                dx = xt[i_aux] - s_xj[j];
                dy = yt[i_aux] - s_yj[j];
                dz = zt[i_aux] - s_zj[j];
                R  = 1/sqrt(dx*dx + dy*dy + dz*dz + eps*eps);
                R2 = R*R;
                R3 = R2*R;
                if (LorY==2) // this if never fires as this function is only used for energy calculations (only laplace)
                {
                    expKr = exp(-kappa*R);
                    dVx_aux[i_aux] += m[j]*expKr*R;
                    dKx_aux[i_aux] += expKr*R2*(kappa+1*R) * (dx*mx[j] + dy*my[j] + dz*mz[j]);
                }
                if (LorY==1)
                {
                    dVx_aux[i_aux] -= m[j]*dx*R3;
                    dVy_aux[i_aux] -= m[j]*dy*R3;
                    dVz_aux[i_aux] -= m[j]*dz*R3;
                    dKx_aux[i_aux] += mx[j]*R3-3*dx*R3*R2*(dx*mx[j] + dy*my[j] + dz*mz[j]);
                    dKy_aux[i_aux] += my[j]*R3-3*dy*R3*R2*(dx*mx[j] + dy*my[j] + dz*mz[j]);
                    dKz_aux[i_aux] += mz[j]*R3-3*dz*R3*R2*(dx*mx[j] + dy*my[j] + dz*mz[j]);
                }
                //stop = get_time();
                //aux[1] += stop - start;
            }
            
            if(condition_an)
            {
                aux[0] += 1;
                REAL center[3] = {xt[i_aux], yt[i_aux], zt[i_aux]};
                REAL panel[9]  = {triangle[9*tri[j]], triangle[9*tri[j]+1], triangle[9*tri[j]+2],
                                triangle[9*tri[j]+3], triangle[9*tri[j]+4], triangle[9*tri[j]+5],
                                triangle[9*tri[j]+6], triangle[9*tri[j]+7], triangle[9*tri[j]+8]};
                REAL dPHI_Kx = 0., dPHI_Ky = 0., dPHI_Kz = 0., dPHI_Vx = 0., dPHI_Vy = 0., dPHI_Vz = 0.;
                
                start = get_time();

                if (same==1) // So far, this if will never fire, as we only use this function for energy calculation (never singular)
                {
                    dPHI_Kx = K_diag;
                    if (IorE==1)
                        dPHI_Vx = sglInt_int[j];
                    else
                        dPHI_Vx = sglInt_ext[j];
                }
                else
                {
                    GQ_fine_derivative(dPHI_Kx, dPHI_Ky, dPHI_Kz, dPHI_Vx, dPHI_Vy, dPHI_Vz, panel, xt[i_aux], yt[i_aux], zt[i_aux], kappa, Xsk, Wsk, WskSize, Area[tri[j]], LorY); 
                }


                stop = get_time();
                aux[1] += stop - start;

//                printf("%f \t %f\n",PHI_V,mVclean[j]);

                dVx_aux[i_aux]  += dPHI_Vx * mVclean[j];
                dVy_aux[i_aux]  += dPHI_Vy * mVclean[j];
                dVz_aux[i_aux]  += dPHI_Vz * mVclean[j];
                dKx_aux[i_aux]  += dPHI_Kx * mKclean[j]; 
                dKy_aux[i_aux]  += dPHI_Ky * mKclean[j]; 
                dKz_aux[i_aux]  += dPHI_Kz * mKclean[j]; 

            }
        }
    }

}


void direct_c_2derivative(REAL *dKxx_aux, int dKxx_auxSize, REAL *dKxy_aux, int dKxy_auxSize, REAL *dKxz_aux, int dKxz_auxSize,
                          REAL *dKyx_aux, int dKyx_auxSize, REAL *dKyy_aux, int dKyy_auxSize, REAL *dKyz_aux, int dKyz_auxSize,
                          REAL *dKzx_aux, int dKzx_auxSize, REAL *dKzy_aux, int dKzy_auxSize, REAL *dKzz_aux, int dKzz_auxSize,
                          REAL *dVxx_aux, int dVxx_auxSize, REAL *dVxy_aux, int dVxy_auxSize, REAL *dVxz_aux, int dVxz_auxSize,
                          REAL *dVyx_aux, int dVyx_auxSize, REAL *dVyy_aux, int dVyy_auxSize, REAL *dVyz_aux, int dVyz_auxSize,
                          REAL *dVzx_aux, int dVzx_auxSize, REAL *dVzy_aux, int dVzy_auxSize, REAL *dVzz_aux, int dVzz_auxSize,
                        int LorY, REAL K_diag, REAL V_diag, int IorE, REAL *triangle, int triangleSize,
                        int *tri, int triSize, int *k, int kSize, REAL *xi, int xiSize, REAL *yi, int yiSize, 
                        REAL *zi, int ziSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, 
                        REAL *s_zj, int s_zjSize, REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
                        REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize, REAL *mKclean, int mKcleanSize, REAL *mVclean, int mVcleanSize,
                        int *target, int targetSize,REAL *Area, int AreaSize, REAL *sglInt_int, int sglInt_intSize, REAL *sglInt_ext, int sglInt_extSize, 
                        REAL *xk, int xkSize, REAL *wk, int wkSize, REAL *Xsk, int XskSize, REAL *Wsk, int WskSize, 
                        REAL kappa, REAL threshold, REAL eps, REAL w0, REAL *aux, int auxSize)
{
    double start,stop;
    int N_target = targetSize;
    int N_source = s_xjSize;
    REAL dx, dy, dz, dx_tri, dy_tri, dz_tri, R, R2, R3, R_tri, expKr;
    bool L_d, same, condition_an, condition_gq;

    for(int i_aux=0; i_aux<N_target; i_aux++)
    {  
        int i = target[i_aux];
        for(int j=0; j<N_source; j++)
        {   
            // Check if panels are far enough for Gauss quadrature
            dx_tri = xt[i_aux] - xi[tri[j]];
            dy_tri = yt[i_aux] - yi[tri[j]];
            dz_tri = zt[i_aux] - zi[tri[j]];
            R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri);
            
            L_d  = (sqrt(2*Area[tri[j]])/(R_tri+eps)>=threshold);
            same = (i==tri[j]);
            condition_an = ((same || L_d) && (k[j]==0));
            condition_gq = (!L_d);

            if(condition_gq)
            {
                //start = get_time();
                dx = xt[i_aux] - s_xj[j];
                dy = yt[i_aux] - s_yj[j];
                dz = zt[i_aux] - s_zj[j];
                R  = 1/sqrt(dx*dx + dy*dy + dz*dz + eps*eps);
                R2 = R*R;
                R3 = R2*R;
                if (LorY==2) // this if never fires as this function is only used for energy calculations (only laplace)
                {
                    expKr = exp(-kappa*R);
                    dVxx_aux[i_aux] += m[j]*expKr*R;
                    dKxx_aux[i_aux] += expKr*R2*(kappa+1*R) * (dx*mx[j] + dy*my[j] + dz*mz[j]);
                }
                if (LorY==1)
                {
                    dVxx_aux[i_aux] += m[j]*R3*(-1 + 3*dx*dx*R2);
                    dVxy_aux[i_aux] += m[j]*R3*3*dx*dy*R2;
                    dVxz_aux[i_aux] += m[j]*R3*3*dx*dz*R2;
                    dVyx_aux[i_aux] += m[j]*R3*3*dy*dx*R2;
                    dVyy_aux[i_aux] += m[j]*R3*(-1 + 3*dy*dy*R2);
                    dVyz_aux[i_aux] += m[j]*R3*3*dy*dz*R2;
                    dVzx_aux[i_aux] += m[j]*R3*3*dz*dx*R2;
                    dVzy_aux[i_aux] += m[j]*R3*3*dz*dy*R2;
                    dVzz_aux[i_aux] += m[j]*R3*(-1 + 3*dz*dz*R2);
                    dKxx_aux[i_aux] -= 3*R3*R2*(2*dx*mx[j]+ dx*mx[j]+dy*my[j]+dz*mz[j] - 5*R2*dx*dx*(dx*mx[j]+dy*my[j]+dz*mz[j]));
                    dKxy_aux[i_aux] -= 3*R3*R2*(dx*my[j] + dy*mx[j] - 5*R2*dx*dy*(dx*mx[j]+dy*my[j]+dz*mz[j]));
                    dKxz_aux[i_aux] -= 3*R3*R2*(dx*mz[j] + dz*mx[j] - 5*R2*dx*dz*(dx*mx[j]+dy*my[j]+dz*mz[j]));
                    dKyx_aux[i_aux] -= 3*R3*R2*(dy*mx[j] + dx*my[j] - 5*R2*dy*dx*(dx*mx[j]+dy*my[j]+dz*mz[j]));
                    dKyy_aux[i_aux] -= 3*R3*R2*(2*dy*my[j]+ dx*mx[j]+dy*my[j]+dz*mz[j] - 5*R2*dy*dy*(dx*mx[j]+dy*my[j]+dz*mz[j]));
                    dKyz_aux[i_aux] -= 3*R3*R2*(dy*mz[j] + dz*my[j] - 5*R2*dy*dz*(dx*mx[j]+dy*my[j]+dz*mz[j]));
                    dKzx_aux[i_aux] -= 3*R3*R2*(dz*mx[j] + dx*mz[j] - 5*R2*dz*dx*(dx*mx[j]+dy*my[j]+dz*mz[j]));
                    dKzy_aux[i_aux] -= 3*R3*R2*(dz*my[j] + dy*mz[j] - 5*R2*dz*dy*(dx*mx[j]+dy*my[j]+dz*mz[j]));
                    dKzz_aux[i_aux] -= 3*R3*R2*(2*dz*mz[j]+ dx*mx[j]+dy*my[j]+dz*mz[j] - 5*R2*dz*dz*(dx*mx[j]+dy*my[j]+dz*mz[j]));
                }
                //stop = get_time();
                //aux[1] += stop - start;
            }
            
            if(condition_an)
            {
                aux[0] += 1;
                REAL center[3] = {xt[i_aux], yt[i_aux], zt[i_aux]};
                REAL panel[9]  = {triangle[9*tri[j]], triangle[9*tri[j]+1], triangle[9*tri[j]+2],
                                triangle[9*tri[j]+3], triangle[9*tri[j]+4], triangle[9*tri[j]+5],
                                triangle[9*tri[j]+6], triangle[9*tri[j]+7], triangle[9*tri[j]+8]};
                REAL dPHI_Kxx = 0., dPHI_Kxy = 0., dPHI_Kxz = 0., 
                     dPHI_Kyx = 0., dPHI_Kyy = 0., dPHI_Kyz = 0.,
                     dPHI_Kzx = 0., dPHI_Kzy = 0., dPHI_Kzz = 0.,
                     dPHI_Vxx = 0., dPHI_Vxy = 0., dPHI_Vxz = 0., 
                     dPHI_Vyx = 0., dPHI_Vyy = 0., dPHI_Vyz = 0.,
                     dPHI_Vzx = 0., dPHI_Vzy = 0., dPHI_Vzz = 0.;
                
                start = get_time();

                if (same==1) // So far, this if will never fire, as we only use this function for energy calculation (never singular)
                {
                    dPHI_Kxx = K_diag;
                    if (IorE==1)
                        dPHI_Vxx = sglInt_int[j];
                    else
                        dPHI_Vxx = sglInt_ext[j];
                }
                else
                {
                    GQ_fine_2derivative(dPHI_Kxx, dPHI_Kxy, dPHI_Kxz, dPHI_Kyx, dPHI_Kyy, dPHI_Kyz,dPHI_Kzx, dPHI_Kzy, dPHI_Kzz,
                                       dPHI_Vxx, dPHI_Vxy, dPHI_Vxz, dPHI_Vyx, dPHI_Vyy, dPHI_Vyz,dPHI_Vzx, dPHI_Vzy, dPHI_Vzz, 
                                       panel, xt[i_aux], yt[i_aux], zt[i_aux], kappa, Xsk, Wsk, WskSize, Area[tri[j]], LorY); 
                }


                stop = get_time();
                aux[1] += stop - start;

//                printf("%f \t %f\n",PHI_V,mVclean[j]);

                dVxx_aux[i_aux]  += dPHI_Vxx * mVclean[j];
                dVxy_aux[i_aux]  += dPHI_Vxy * mVclean[j];
                dVxz_aux[i_aux]  += dPHI_Vxz * mVclean[j];
                dVyx_aux[i_aux]  += dPHI_Vyx * mVclean[j];
                dVyy_aux[i_aux]  += dPHI_Vyy * mVclean[j];
                dVyz_aux[i_aux]  += dPHI_Vyz * mVclean[j];
                dVzx_aux[i_aux]  += dPHI_Vzx * mVclean[j];
                dVzy_aux[i_aux]  += dPHI_Vzy * mVclean[j];
                dVzz_aux[i_aux]  += dPHI_Vzz * mVclean[j];

                dKxx_aux[i_aux]  += dPHI_Kxx * mKclean[j];
                dKxy_aux[i_aux]  += dPHI_Kxy * mKclean[j];
                dKxz_aux[i_aux]  += dPHI_Kxz * mKclean[j];
                dKyx_aux[i_aux]  += dPHI_Kyx * mKclean[j];
                dKyy_aux[i_aux]  += dPHI_Kyy * mKclean[j];
                dKyz_aux[i_aux]  += dPHI_Kyz * mKclean[j];
                dKzx_aux[i_aux]  += dPHI_Kzx * mKclean[j];
                dKzy_aux[i_aux]  += dPHI_Kzy * mKclean[j];
                dKzz_aux[i_aux]  += dPHI_Kzz * mKclean[j];
            }
        }
    }

}

void direct_sort(REAL *K_aux, int K_auxSize, REAL *V_aux, int V_auxSize, int LorY, REAL K_diag, REAL V_diag, int IorE, REAL *triangle, int triangleSize,
        int *tri, int triSize, int *k, int kSize, REAL *xi, int xiSize, REAL *yi, int yiSize, 
        REAL *zi, int ziSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, 
        REAL *s_zj, int s_zjSize, REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize, REAL *mKclean, int mKcleanSize, REAL *mVclean, int mVcleanSize,
        int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize, int *offSrc, int offSrcSize, int *offTwg, int offTwgSize,  
        int *target, int targetSize,REAL *Area, int AreaSize, REAL *sglInt_int, int sglInt_intSize, REAL *sglInt_ext, int sglInt_extSize, 
        REAL *xk, int xkSize, REAL *wk, int wkSize, REAL *Xsk, int XskSize, REAL *Wsk, int WskSize,
        REAL kappa, REAL threshold, REAL eps, REAL w0, REAL *aux, int auxSize)
{
    double start,stop;
    int CI_start, CI_end, CJ_start, CJ_end, list_start, list_end, CJ;
    REAL dx, dy, dz, dx_tri, dy_tri, dz_tri, R, R2, R3, R_tri, expKr, sum_K, sum_V;
    bool L_d, same, condition_an, condition_gq;

    for (int tarTwg=0; tarTwg<offTarSize; tarTwg++)
    {
        CI_start = offTar[tarTwg];
        CI_end   = offTar[tarTwg] + sizeTar[tarTwg];
        list_start = offTwg[tarTwg];
        list_end   = offTwg[tarTwg+1];

        for(int i=CI_start; i<CI_end; i++)
        {  
            sum_K = 0.;
            sum_V = 0.;

            for (int lst=list_start; lst<list_end; lst++)
            {
                CJ = interList[lst];
                CJ_start = offSrc[CJ];
                CJ_end = offSrc[CJ+1];

                for(int j=CJ_start; j<CJ_end; j++)
                {   
                    // Check if panels are far enough for Gauss quadrature
                    //start = get_time();
                    int ptr = 9*j;
                    REAL panel[9]  = {triangle[ptr], triangle[ptr+1], triangle[ptr+2],
                                    triangle[ptr+3], triangle[ptr+4], triangle[ptr+5],
                                    triangle[ptr+6], triangle[ptr+7], triangle[ptr+8]};

                    dx_tri = xt[i] - (panel[0]+panel[3]+panel[6])/3;
                    dy_tri = yt[i] - (panel[1]+panel[4]+panel[7])/3;
                    dz_tri = zt[i] - (panel[2]+panel[5]+panel[8])/3;
                    R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri);
                    
                    L_d  = (sqrt(2*Area[j])/(R_tri+eps)>=threshold);
                    same = (R_tri<1e-12);
                    condition_an = ((L_d) && (k[j]==0));
                    condition_gq = (!L_d);
                    //stop = get_time();
                    //aux[1] += stop - start;

                    if(condition_gq)
                    {
                        //start = get_time();
                        dx = xt[i] - s_xj[j];
                        dy = yt[i] - s_yj[j];
                        dz = zt[i] - s_zj[j];
                        R  = sqrt(dx*dx + dy*dy + dz*dz + eps*eps);
                        R2 = R*R;
                        R3 = R2*R;
                        if (LorY==2)
                        {
                            expKr = exp(-kappa*R);
                            sum_V += m[j]*expKr/R;
                            sum_K += expKr/R2*(kappa+1/R) * (dx*mx[j] + dy*my[j] + dz*mz[j]);
                        }
                        if (LorY==1)
                        {
                            sum_V += m[j]/R;
                            sum_K += 1/R3*(dx*mx[j] + dy*my[j] + dz*mz[j]);
                        }
                        //stop = get_time();
                        //aux[1] += stop - start;
                    }
                    
                    if(condition_an)
                    {
                        start = get_time();
                        aux[0] += 1;
                        REAL center[3] = {xt[i], yt[i], zt[i]};
                        REAL PHI_K = 0., PHI_V = 0.;
                        
                        if (same==1)
                        {
                            PHI_K = K_diag;
                            if (IorE==1)
                                PHI_V = sglInt_int[j];
                            else
                                PHI_V = sglInt_ext[j];
                        }
                        else
                        {
                            GQ_fine(PHI_K, PHI_V, panel, xt[i], yt[i], zt[i], kappa, Xsk, Wsk, WskSize, Area[j], LorY); 
                        }

        //                printf("%f \t %f\n",PHI_V,mVclean[j]);

                        sum_V += PHI_V * mVclean[j];
                        sum_K += PHI_K * mKclean[j]; 
                        stop = get_time();
                        aux[1] += stop - start;

                    }
                }
            }

            V_aux[i] += sum_V;
            K_aux[i] += sum_K;
        }
    }
}


void directKt_sort(REAL *Ktx_aux, int Ktx_auxSize, REAL *Kty_aux, int Kty_auxSize, REAL *Ktz_aux, int Ktz_auxSize, 
        int LorY, REAL *triangle, int triangleSize,
        int *k, int kSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, REAL *s_zj, int s_zjSize, 
        REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        REAL *m, int mSize, REAL *mKclean, int mKcleanSize,
        int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize, 
        int *offSrc, int offSrcSize, int *offTwg, int offTwgSize, REAL *Area, int AreaSize,
        REAL *Xsk, int XskSize, REAL *Wsk, int WskSize, REAL kappa, REAL threshold, REAL eps, REAL *aux, int auxSize)
{
    double start,stop;
    int CI_start, CI_end, CJ_start, CJ_end, list_start, list_end, CJ;
    REAL dx, dy, dz, dx_tri, dy_tri, dz_tri, R, R2, R3, R_tri, expKr, sum_Ktx, sum_Kty, sum_Ktz;
    bool L_d, same, condition_an, condition_gq;

    for (int tarTwg=0; tarTwg<offTarSize; tarTwg++)
    {
        CI_start = offTar[tarTwg];
        CI_end   = offTar[tarTwg] + sizeTar[tarTwg];
        list_start = offTwg[tarTwg];
        list_end   = offTwg[tarTwg+1];

        for(int i=CI_start; i<CI_end; i++)
        {  
            sum_Ktx = 0.;
            sum_Kty = 0.;
            sum_Ktz = 0.;

            for (int lst=list_start; lst<list_end; lst++)
            {
                CJ = interList[lst];
                CJ_start = offSrc[CJ];
                CJ_end = offSrc[CJ+1];

                for(int j=CJ_start; j<CJ_end; j++)
                {   
                    // Check if panels are far enough for Gauss quadrature
                    //start = get_time();
                    int ptr = 9*j;
                    REAL panel[9]  = {triangle[ptr], triangle[ptr+1], triangle[ptr+2],
                                    triangle[ptr+3], triangle[ptr+4], triangle[ptr+5],
                                    triangle[ptr+6], triangle[ptr+7], triangle[ptr+8]};

                    dx_tri = xt[i] - (panel[0]+panel[3]+panel[6])/3;
                    dy_tri = yt[i] - (panel[1]+panel[4]+panel[7])/3;
                    dz_tri = zt[i] - (panel[2]+panel[5]+panel[8])/3;
                    R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri);
                    
                    L_d  = (sqrt(2*Area[j])/(R_tri+eps)>=threshold);
                    same = (R_tri<1e-12);
                    condition_an = ((L_d) && (k[j]==0));
                    condition_gq = (!L_d);
                    //stop = get_time();
                    //aux[1] += stop - start;

                    if(condition_gq)
                    {
                        //start = get_time();
                        dx = xt[i] - s_xj[j];
                        dy = yt[i] - s_yj[j];
                        dz = zt[i] - s_zj[j];
                        R  = sqrt(dx*dx + dy*dy + dz*dz + eps*eps);
                        R2 = R*R;
                        R3 = R2*R;
                        if (LorY==2)
                        {
                            expKr = m[j]*exp(-kappa*R)/R2*(kappa+1/R);
                            sum_Ktx -= expKr * dx;
                            sum_Kty -= expKr * dy;
                            sum_Ktz -= expKr * dz;
                        }
                        if (LorY==1)
                        {
                            expKr = m[j]/R3;
                            sum_Ktx -= expKr*dx;
                            sum_Kty -= expKr*dy;
                            sum_Ktz -= expKr*dz;
                        }
                        //stop = get_time();
                        //aux[1] += stop - start;
                    }
                    
                    if(condition_an)
                    {
                        start = get_time();
                        aux[0] += 1;
                        REAL PHI_Ktx = 0.;
                        REAL PHI_Kty = 0.;
                        REAL PHI_Ktz = 0.;
                        
                        if (same==1)
                        {
                            PHI_Ktx = 0;
                            PHI_Kty = 0;
                            PHI_Ktz = 0;
                        }
                        else
                        {
                            GQ_fineKt(PHI_Ktx, PHI_Kty, PHI_Ktz, panel, xt[i], yt[i], zt[i], kappa, Xsk, Wsk, WskSize, Area[j], LorY); 
                        }

        //                printf("%f \t %f\n",PHI_V,mVclean[j]);

                        sum_Ktx += PHI_Ktx * mKclean[j]; 
                        sum_Kty += PHI_Kty * mKclean[j]; 
                        sum_Ktz += PHI_Ktz * mKclean[j]; 
                        stop = get_time();
                        aux[1] += stop - start;

                    }
                }
            }

            Ktx_aux[i] += sum_Ktx;
            Kty_aux[i] += sum_Kty;
            Ktz_aux[i] += sum_Ktz;
        }
    }
}

void coulomb_direct(REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize, 
                    REAL *m, int mSize, REAL *K_aux, int K_auxSize)
{
    REAL sum, dx, dy, dz, r;
    for(int i=0; i<xtSize; i++)
    {
        sum = 0.;
        for(int j=0; j<xtSize; j++)
        {
            if (i!=j)
            {
                dx = xt[i] - xt[j];
                dy = yt[i] - yt[j];
                dz = zt[i] - zt[j];
                r  = sqrt(dx*dx + dy*dy + dz*dz);
                sum += m[j]/r;
            }
        }
        K_aux[i] = m[i]*sum;
    }
}

void coulomb_phi_multipole(REAL *xt, REAL *yt, REAL *zt, REAL *q, REAL *px, REAL *py, REAL *pz,
                        REAL *Qxx, REAL *Qxy, REAL *Qxz, REAL *Qyx, REAL *Qyy, REAL *Qyz,
                        REAL *Qzx, REAL *Qzy, REAL *Qzz, REAL *phi, int N)
{
    REAL Ri[3], Rnorm, R3, R5, sum;
    REAL T0, T1[3], T2[3][3];
    REAL eps = 1e-15;

    for (int i=0; i<N; i++)
    {
        sum = 0.0;
        for (int j=0; j<N; j++)
        {
            Ri[0] = xt[i] - xt[j]; 
            Ri[1] = yt[i] - yt[j]; 
            Ri[2] = zt[i] - zt[j]; 
            Rnorm = sqrt(Ri[0]*Ri[0]+Ri[1]*Ri[1]+Ri[2]*Ri[2]+eps*eps);
            R3 = Rnorm*Rnorm*Rnorm;
            R5 = R3*Rnorm*Rnorm;
            
            if (Rnorm>1e-12) //remove singularity
            {
                T0 = 1/Rnorm;
                for (int k=0; k<3; k++)
                {
                    T1[k] = Ri[k]/R3;
                    for (int l=0; l<3; l++)
                    {
                        T2[k][l] = Ri[k]*Ri[l]/R5;   
                    }
                }                          
                sum += T0*q[j] + T1[0]*px[j] + T1[1]*py[j] + T1[2]*pz[j] 
                + 0.5*(T2[0][0]*Qxx[j] + T2[0][1]*Qxy[j] + T2[0][2]*Qxz[j]  
                     + T2[1][0]*Qyx[j] + T2[1][1]*Qyy[j] + T2[1][2]*Qyz[j]  
                     + T2[2][0]*Qzx[j] + T2[2][1]*Qzy[j] + T2[2][2]*Qzz[j]);
            }
        }
        phi[i] += sum;
    }
}


void coulomb_dphi_multipole(REAL *xt, REAL *yt, REAL *zt, REAL *q, REAL *px, REAL *py, REAL *pz,
                        REAL *Qxx, REAL *Qxy, REAL *Qxz, REAL *Qyx, REAL *Qyy, REAL *Qyz,
                        REAL *Qzx, REAL *Qzy, REAL *Qzz,  REAL *alpha, REAL *thole, int *polar_group, bool flag_polar_group, 
                        REAL dphi[][3], int N)
{
    REAL Ri[3], Rnorm, R3, R5, R7, sum[3];
    REAL T0, T1[3], T2[3][3];
    REAL dkl, dkm;
    REAL eps = 1e-15;
    bool not_same_polar_group;
    REAL scale3 = 1.0;
    REAL scale5 = 1.0;
    REAL scale7 = 1.0;
    REAL damp, gamma, expdamp;

    for (int i=0; i<N; i++)
    {
        sum[0] =  0.0;
        sum[1] =  0.0;
        sum[2] =  0.0;
        for (int j=0; j<N; j++)
        {
            Ri[0] = xt[i] - xt[j]; 
            Ri[1] = yt[i] - yt[j]; 
            Ri[2] = zt[i] - zt[j]; 
            Rnorm = sqrt(Ri[0]*Ri[0]+Ri[1]*Ri[1]+Ri[2]*Ri[2]+eps*eps);
            R3 = Rnorm*Rnorm*Rnorm;
            R5 = R3*Rnorm*Rnorm;
            R7 = R5*Rnorm*Rnorm;
            
            if (flag_polar_group==false)
            {
                not_same_polar_group = true;
            }
            else
            {
                gamma = std::min(thole[i], thole[j]);
                damp = pow(alpha[i]*alpha[j],0.16666667);
                damp += 1e-12;
                damp = -gamma*(R3/(damp*damp*damp));
                expdamp = exp(damp);
                scale3 = 1 - expdamp;
                scale5 = 1 - expdamp*(1-damp);
                scale7 = 1 - expdamp*(1-damp+0.6*damp*damp);
                if (polar_group[i]!=polar_group[j])
                {
                    not_same_polar_group = true;
                }
                else
                {
                    not_same_polar_group = false;
                }
            }

            if ((Rnorm>1e-12) && (not_same_polar_group==true)) //remove singularity
            {
                for (int k=0; k<3; k++)
                {
                    T0 = -Ri[k]/R3 * scale3;
                    for (int l=0; l<3; l++)
                    {
                        dkl = (REAL)(k==l);
                        T1[l] = dkl/R3 * scale3 - 3*Ri[k]*Ri[l]/R5 * scale5;
                        for (int m=0; m<3; m++)
                        {
                            dkm = (REAL)(k==m);
                            T2[l][m] = (dkm*Ri[l]+dkl*Ri[m])/R5 * scale5 - 5*Ri[l]*Ri[m]*Ri[k]/R7 * scale7;
                        }
                    }

                    sum[k] += T0*q[j] + T1[0]*px[j] + T1[1]*py[j] + T1[2]*pz[j] 
                           + 0.5*(T2[0][0]*Qxx[j] + T2[0][1]*Qxy[j] + T2[0][2]*Qxz[j]   
                                + T2[1][0]*Qyx[j] + T2[1][1]*Qyy[j] + T2[1][2]*Qyz[j]  
                                + T2[2][0]*Qzx[j] + T2[2][1]*Qzy[j] + T2[2][2]*Qzz[j]);
                }                          
            }
            
           
        }
        dphi[i][0] += sum[0];
        dphi[i][1] += sum[1];
        dphi[i][2] += sum[2];
    }
}

void coulomb_ddphi_multipole(REAL *xt, REAL *yt, REAL *zt, REAL *q, REAL *px, REAL *py, REAL *pz,
                        REAL *Qxx, REAL *Qxy, REAL *Qxz, REAL *Qyx, REAL *Qyy, REAL *Qyz,
                        REAL *Qzx, REAL *Qzy, REAL *Qzz, REAL ddphi[][3][3], int N)
{
    REAL Ri[3], Rnorm, R3, R5, R7, R9, sum[3][3];
    REAL T0, T1[3], T2[3][3];
    REAL dkl, dkm, dlm, dkn, dln;
    REAL eps = 1e-15;

    for (int i=0; i<N; i++)
    {
        sum[0][0] = 0.0;
        sum[0][1] = 0.0;
        sum[0][2] = 0.0;
        sum[1][0] = 0.0;
        sum[1][1] = 0.0;
        sum[1][2] = 0.0;
        sum[2][0] = 0.0;
        sum[2][1] = 0.0;
        sum[2][2] = 0.0;
        for (int j=0; j<N; j++)
        {
            Ri[0] = xt[i] - xt[j]; 
            Ri[1] = yt[i] - yt[j]; 
            Ri[2] = zt[i] - zt[j]; 
            Rnorm = sqrt(Ri[0]*Ri[0]+Ri[1]*Ri[1]+Ri[2]*Ri[2]+eps*eps);
            R3 = Rnorm*Rnorm*Rnorm;
            R5 = R3*Rnorm*Rnorm;
            R7 = R5*Rnorm*Rnorm;
            R9 = R3*R3*R3;
            
            if (Rnorm>1e-12) //remove singularity
            {
                for (int k=0; k<3; k++)
                {
                    for (int l=0; l<3; l++)
                    {
                        dkl = (REAL)(k==l);
                        T0 = -dkl/R3 + 3*Ri[k]*Ri[l]/R5;

                        for (int m=0; m<3; m++)
                        {
                            dkm = (REAL)(k==m);
                            dlm = (REAL)(l==m);
                            T1[m] = -3*(dkm*Ri[l]+dkl*Ri[m]+dlm*Ri[k])/R5 + 15*Ri[l]*Ri[m]*Ri[k]/R7;
                                                    
                            for (int n=0; n<3; n++)
                            {
                                dkn = (REAL)(k==n);
                                dln = (REAL)(l==n);
                                T2[m][n] = 35*Ri[k]*Ri[l]*Ri[m]*Ri[n]/R9
                                        - 5*(Ri[m]*Ri[n]*dkl + Ri[l]*Ri[n]*dkm
                                           + Ri[m]*Ri[l]*dkn + Ri[k]*Ri[n]*dlm
                                           + Ri[m]*Ri[k]*dln)/R7
                                        + (dkm*dln + dlm*dkn)/R5;
 
                            }
                        }
                        sum[k][l] += T0*q[j] + T1[0]*px[j] + T1[1]*py[j] + T1[2]*pz[j] 
                               + 0.5*(T2[0][0]*Qxx[j] + T2[0][1]*Qxy[j] + T2[0][2]*Qxz[j]  
                                    + T2[1][0]*Qyx[j] + T2[1][1]*Qyy[j] + T2[1][2]*Qyz[j]  
                                    + T2[2][0]*Qzx[j] + T2[2][1]*Qzy[j] + T2[2][2]*Qzz[j]);
                    }
                }                          
            }
        }
        for (int k=0; k<3; k++)
        {
            for (int l=0; l<3; l++)
            {
                ddphi[i][k][l] += sum[k][l];
            }
        }
    }
}

void coulomb_phi_multipole_Thole(REAL *xt, REAL *yt, REAL *zt, REAL *px, REAL *py, REAL *pz, 
                                  REAL *alpha, REAL *thole, int *polar_group, 
                                  int *connections_12, int *pointer_connections_12,
                                  int *connections_13, int *pointer_connections_13,
                                  REAL p12scale, REAL p13scale, REAL *phi, int N)
{
	REAL r, r3, r5;
	REAL eps = 1e-15;
	REAL T1[3], Ri[3], sum;
	REAL dkl, dkm;
    REAL scale3 = 1.0;
    REAL damp, gamma, expdamp;
    int start_12, stop_12, start_13, stop_13;
    REAL pscale;


    for (int i=0; i<N; i++)
    {
        sum = 0.0;
        start_12 = pointer_connections_12[i]; 
        stop_12 = pointer_connections_12[i+1]; 
        start_13 = pointer_connections_13[i]; 
        stop_13 = pointer_connections_13[i+1]; 
        for (int j=0; j<N; j++)
        {
            pscale = 1.0;
            for (int ii=start_12; ii<stop_12; ii++)
            {
                if (connections_12[ii]==j)
                {
                    pscale = p12scale;
                }
            }

            for (int ii=start_13; ii<stop_13; ii++)
            {
                if (connections_13[ii]==j)
                {
                    pscale = p13scale;
                }
            }

            Ri[0] = xt[i] - xt[j]; 
            Ri[1] = yt[i] - yt[j]; 
            Ri[2] = zt[i] - zt[j]; 

            r  = 1./sqrt(Ri[0]*Ri[0] + Ri[1]*Ri[1] + Ri[2]*Ri[2] + eps*eps);
            r3 = r*r*r;
            
            gamma = std::min(thole[i], thole[j]);
            damp = pow(alpha[i]*alpha[j],0.16666667);
            damp += 1e-12;
            damp = -gamma*(1/(r3*damp*damp*damp));
            expdamp = exp(damp);
            scale3 = 1 - expdamp;

            if (r<1e12) //remove singularity
            {
                for (int k=0; k<3; k++)
                {
                    T1[k] = Ri[k]*r3*scale3*pscale;
                }                          
                sum += T1[0]*px[j] + T1[1]*py[j] + T1[2]*pz[j];
            }
        }

        phi[i] += sum;
    }

}

void coulomb_dphi_multipole_Thole(REAL *xt, REAL *yt, REAL *zt, REAL *px, REAL *py, REAL *pz, 
                                  REAL *alpha, REAL *thole, int *polar_group, 
                                  int *connections_12, int *pointer_connections_12,
                                  int *connections_13, int *pointer_connections_13,
                                  REAL p12scale, REAL p13scale, REAL dphi[][3], int N)
{
	REAL r, r3, r5;
	REAL eps = 1e-15;
	REAL T1[3], Ri[3], sum[3];
	REAL dkl, dkm;
    REAL scale3 = 1.0;
    REAL scale5 = 1.0;
    REAL damp, gamma, expdamp;
    int start_12, stop_12, start_13, stop_13;
    REAL pscale;

    for (int i=0; i<N; i++)
    {
        sum[0] = 0;
        sum[1] = 0;
        sum[2] = 0;
        start_12 = pointer_connections_12[i]; 
        stop_12 = pointer_connections_12[i+1]; 
        start_13 = pointer_connections_13[i]; 
        stop_13 = pointer_connections_13[i+1]; 

        for (int j=0; j<N; j++)
        {
            pscale = 1.0;
            for (int ii=start_12; ii<stop_12; ii++)
            {
                if (connections_12[ii]==j)
                {
                    pscale = p12scale;
                }
            }

            for (int ii=start_13; ii<stop_13; ii++)
            {
                if (connections_13[ii]==j)
                {
                    pscale = p13scale;
                }
            }

            Ri[0] = xt[i] - xt[j];
            Ri[1] = yt[i] - yt[j];
            Ri[2] = zt[i] - zt[j];

            r  = 1./sqrt(Ri[0]*Ri[0] + Ri[1]*Ri[1] + Ri[2]*Ri[2] + eps*eps);
            r3 = r*r*r;
            r5 = r3*r*r;

            gamma = std::min(thole[i], thole[j]);
            damp = pow(alpha[i]*alpha[j],0.16666667);
            damp += 1e-12;
            damp = -gamma*(1/(r3*damp*damp*damp));
            expdamp = exp(damp);
            scale3 = 1 - expdamp;
            scale5 = 1 - expdamp*(1-damp);

            if (r<1e12) // Remove singularity
            {
                for (int k=0; k<3; k++)
                {
                    for (int l=0; l<3; l++)
                    {
                        dkl = (REAL)(k==l);
                        T1[l] = scale3*dkl*r3*pscale - scale5*3*Ri[k]*Ri[l]*r5*pscale;
                    }

                    sum[k] += T1[0]*px[j] + T1[1]*py[j] + T1[2]*pz[j];
                }
            }
        }
        dphi[i][0] += sum[0];
        dphi[i][1] += sum[1];
        dphi[i][2] += sum[2];
    }
}

void coulomb_ddphi_multipole_Thole(REAL *xt, REAL *yt, REAL *zt, REAL *px, REAL *py, REAL *pz, 
                                  REAL *alpha, REAL *thole, int *polar_group, 
                                  int *connections_12, int *pointer_connections_12,
                                  int *connections_13, int *pointer_connections_13,
                                  REAL p12scale, REAL p13scale, REAL ddphi[][3][3], int N)
{
	REAL r, r3, r5, r7;
	REAL eps = 1e-15;
	REAL T1[3], Ri[3], sum[3][3];
	REAL dkl, dkm, dlm;
    REAL scale3 = 1.0;
    REAL scale5 = 1.0;
    REAL scale7 = 1.0;
    REAL damp, gamma, expdamp;
    int start_12, stop_12, start_13, stop_13;
    REAL pscale; 

    for (int i=0; i<N; i++)
    {
        sum[0][0] = 0.0;
        sum[0][1] = 0.0;
        sum[0][2] = 0.0;
        sum[1][0] = 0.0;
        sum[1][1] = 0.0;
        sum[1][2] = 0.0;
        sum[2][0] = 0.0;
        sum[2][1] = 0.0;
        sum[2][2] = 0.0;

        start_12 = pointer_connections_12[i]; 
        stop_12 = pointer_connections_12[i+1]; 
        start_13 = pointer_connections_13[i]; 
        stop_13 = pointer_connections_13[i+1]; 

        for (int j=0; j<N; j++)
        {
            pscale = 1.0;
            for (int ii=start_12; ii<stop_12; ii++)
            {
                if (connections_12[ii]==j)
                {
                    pscale = p12scale;
                }
            }

            for (int ii=start_13; ii<stop_13; ii++)
            {
                if (connections_13[ii]==j)
                {
                    pscale = p13scale;
                }
            }

            Ri[0] = xt[i] - xt[j]; 
            Ri[1] = yt[i] - yt[j]; 
            Ri[2] = zt[i] - zt[j]; 

            r  = 1./sqrt(Ri[0]*Ri[0] + Ri[1]*Ri[1] + Ri[2]*Ri[2] + eps*eps);
            r3 = r*r*r;
            r5 = r3*r*r;
            r7 = r5*r*r;

            gamma = std::min(thole[i], thole[j]);
            damp = pow(alpha[i]*alpha[j],0.16666667);
            damp += 1e-12;
            damp = -gamma*(1/(r3*damp*damp*damp));
            expdamp = exp(damp);
            scale5 = 1 - expdamp*(1-damp);
            scale7 = 1 - expdamp*(1-damp+0.6*damp*damp);
            
            if (r<1e12) //remove singularity
            {
                for (int k=0; k<3; k++)
                {
                    for (int l=0; l<3; l++)
                    {
                        dkl = (REAL)(k==l);

                        for (int m=0; m<3; m++)
                        {
                            dkm = (REAL)(k==m);
                            dlm = (REAL)(l==m);
                            T1[m] = -3*(dkm*Ri[l]+dkl*Ri[m]+dlm*Ri[k])*r5*scale5*pscale 
                                    + 15*Ri[l]*Ri[m]*Ri[k]*r7*scale7*pscale;
                                                    
                        }
                        sum[k][l] += T1[0]*px[j] + T1[1]*py[j] + T1[2]*pz[j];
                    }
                }                          
            }
        }

        for (int k=0; k<3; k++)
        {
            for (int l=0; l<3; l++)
            {
                ddphi[i][k][l] += sum[k][l];
            }
        }
    }
}


void coulomb_energy_multipole(REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize, 
                        REAL *q, int qSize, REAL *px, int pxSize, REAL *py, int pySize, REAL *pz, int pzSize, 
                        REAL *px_pol, int px_polSize, REAL *py_pol, int py_polSize, REAL *pz_pol, int pz_polSize, 
                        REAL *Qxx, int QxxSize, REAL *Qxy, int QxySize, REAL *Qxz, int QxzSize, 
                        REAL *Qyx, int QyxSize, REAL *Qyy, int QyySize, REAL *Qyz, int QyzSize, 
                        REAL *Qzx, int QzxSize, REAL *Qzy, int QzySize, REAL *Qzz, int QzzSize, 
                        REAL *alphaxx, int alphaxxSize, REAL *thole, int tholeSize, 
                        int *polar_group, int polar_groupSize, 
                        int *connections_12, int connections_12Size, 
                        int *pointer_connections_12, int pointer_connections_12Size,
                        int *connections_13, int connections_13Size, 
                        int *pointer_connections_13, int pointer_connections_13Size,
                        REAL p12scale, REAL p13scale, REAL *K_aux, int K_auxSize)
{
    double phi  [xtSize];
    double dphi [xtSize][3];
    double ddphi[xtSize][3][3];
    double px_tot[xtSize], py_tot[xtSize], pz_tot[xtSize];
    
    for (int i=0; i<xtSize; i++)
    {
        px_tot[i] = px[i] + px_pol[i];
        py_tot[i] = py[i] + py_pol[i];
        pz_tot[i] = pz[i] + pz_pol[i];
        phi[i] = 0.0;
        dphi[i][0] =  0.0;
        dphi[i][1] =  0.0;
        dphi[i][2] =  0.0;
        ddphi[i][0][0] = 0.0;
        ddphi[i][0][1] = 0.0;
        ddphi[i][0][2] = 0.0;
        ddphi[i][1][0] = 0.0;
        ddphi[i][1][1] = 0.0;
        ddphi[i][1][2] = 0.0;
        ddphi[i][2][0] = 0.0;
        ddphi[i][2][1] = 0.0;
        ddphi[i][2][2] = 0.0;
    }

    bool flag_polar_group = false;
    int *dummy;
    double *dummy2;

    // phi, dphi and ddphi from permanent multipoles
    coulomb_phi_multipole(xt, yt, zt, q, px, py, pz,
                           Qxx, Qxy, Qxz, Qyx, Qyy, Qyz,
                           Qzx, Qzy, Qzz, phi, xtSize);

    coulomb_dphi_multipole(xt, yt, zt, q, px, py, pz,
                           Qxx, Qxy, Qxz, Qyx, Qyy, Qyz,
                           Qzx, Qzy, Qzz, dummy2, dummy2, dummy, flag_polar_group, dphi, xtSize);

    coulomb_ddphi_multipole(xt, yt, zt, q, px, py, pz,
                           Qxx, Qxy, Qxz, Qyx, Qyy, Qyz,
                           Qzx, Qzy, Qzz, ddphi, xtSize);


    // phi, dphi and ddphi from induced dipoles

    coulomb_phi_multipole_Thole(xt, yt, zt, px_pol, py_pol, pz_pol, alphaxx, thole,
                                polar_group, connections_12, pointer_connections_12, 
                                connections_13, pointer_connections_13, 
                                p12scale, p13scale, phi, xtSize);

    coulomb_dphi_multipole_Thole(xt, yt, zt, px_pol, py_pol, pz_pol, alphaxx, thole,
                                  polar_group, connections_12, pointer_connections_12, 
                                  connections_13, pointer_connections_13, 
                                  p12scale, p13scale, dphi, xtSize);

    coulomb_ddphi_multipole_Thole(xt, yt, zt, px_pol, py_pol, pz_pol, alphaxx, thole, 
                                  polar_group, connections_12, pointer_connections_12, 
                                  connections_13, pointer_connections_13, 
                                  p12scale, p13scale, ddphi, xtSize);


    for (int i=0; i<xtSize; i++)
    {
        K_aux[i] = (q[i]*phi[i] 
                  + px[i]*dphi[i][0] + py[i]*dphi[i][1] + pz[i]*dphi[i][2]
                  +(Qxx[i]*ddphi[i][0][0] + Qxy[i]*ddphi[i][0][1] + Qxz[i]*ddphi[i][0][2] 
                  + Qxy[i]*ddphi[i][1][0] + Qyy[i]*ddphi[i][1][1] + Qyz[i]*ddphi[i][1][2] 
                  + Qxz[i]*ddphi[i][2][0] + Qzy[i]*ddphi[i][2][1] + Qzz[i]*ddphi[i][2][2])/6.); 
                    // Energy calculated with p (rather than p_tot) to account for polarization energy
    }
    
}

void compute_induced_dipole(REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize, 
                        REAL *q, int qSize, REAL *px, int pxSize, REAL *py, int pySize, REAL *pz, int pzSize, 
                        REAL *px_pol, int px_polSize, REAL *py_pol, int py_polSize, REAL *pz_pol, int pz_polSize, 
                        REAL *Qxx, int QxxSize, REAL *Qxy, int QxySize, REAL *Qxz, int QxzSize, 
                        REAL *Qyx, int QyxSize, REAL *Qyy, int QyySize, REAL *Qyz, int QyzSize, 
                        REAL *Qzx, int QzxSize, REAL *Qzy, int QzySize, REAL *Qzz, int QzzSize, 
                        REAL *alphaxx, int alphaxxSize, REAL *alphaxy, int alphaxySize, REAL *alphaxz, int alphaxzSize, 
                        REAL *alphayx, int alphayxSize, REAL *alphayy, int alphayySize, REAL *alphayz, int alphayzSize, 
                        REAL *alphazx, int alphazxSize, REAL *alphazy, int alphazySize, REAL *alphazz, int alphazzSize, 
                        REAL *thole, int tholeSize, int *polar_group, int polar_groupSize, 
                        int *connections_12, int connections_12Size, 
                        int *pointer_connections_12, int pointer_connections_12Size,
                        int *connections_13, int connections_13Size, 
                        int *pointer_connections_13, int pointer_connections_13Size,
                        REAL *dphix_reac, int dphix_reacSize, 
                        REAL *dphiy_reac, int dphiy_reacSize, REAL *dphiz_reac, int dphiz_reacSize, double E)
{
    double dphi_coul [xtSize][3];
    double u12scale = 1.0; // mutual-12-scale
    double u13scale = 1.0; // mutual-13-scale
    
    for (int i=0; i<xtSize; i++)
    {
        dphi_coul[i][0] =  0.0;
        dphi_coul[i][1] =  0.0;
        dphi_coul[i][2] =  0.0;
    }

    bool flag_polar_group = true; 

    // Compute derivative of phi (negative of electric field)
    coulomb_dphi_multipole(xt, yt, zt, q, px, py, pz,
                           Qxx, Qxy, Qxz, Qyx, Qyy, Qyz, Qzx, Qzy, Qzz, alphaxx, 
                           thole, polar_group, flag_polar_group, dphi_coul, xtSize);

    coulomb_dphi_multipole_Thole(xt, yt, zt, px_pol, py_pol, pz_pol, alphaxx, thole,
                                  polar_group, connections_12, pointer_connections_12, 
                                  connections_13, pointer_connections_13, 
                                  u12scale, u13scale, dphi_coul, xtSize);

    double SOR = 0.7;
    for (int i=0; i<xtSize; i++)
    {
        //px_pol[i] = (-alphaxx[i]*(dphi_coul[i][0]/(4*M_PI*E)+dphix_reac[i]) 
        //             -alphaxy[i]*(dphi_coul[i][1]/(4*M_PI*E)+dphiy_reac[i]) 
        //             -alphaxz[i]*(dphi_coul[i][2]/(4*M_PI*E)+dphiz_reac[i]));
        //py_pol[i] = (-alphayx[i]*(dphi_coul[i][0]/(4*M_PI*E)+dphix_reac[i]) 
        //             -alphayy[i]*(dphi_coul[i][1]/(4*M_PI*E)+dphiy_reac[i]) 
        //             -alphayz[i]*(dphi_coul[i][2]/(4*M_PI*E)+dphiz_reac[i]));
        //pz_pol[i] = (-alphazx[i]*(dphi_coul[i][0]/(4*M_PI*E)+dphix_reac[i]) 
        //             -alphazy[i]*(dphi_coul[i][1]/(4*M_PI*E)+dphiy_reac[i]) 
        //             -alphazz[i]*(dphi_coul[i][2]/(4*M_PI*E)+dphiz_reac[i]));
        px_pol[i] = px_pol[i]*(1-SOR) + 
                    (-alphaxx[i]*(dphi_coul[i][0]/E+4*M_PI*dphix_reac[i]) 
                     -alphaxy[i]*(dphi_coul[i][1]/E+4*M_PI*dphiy_reac[i]) 
                     -alphaxz[i]*(dphi_coul[i][2]/E+4*M_PI*dphiz_reac[i]))*SOR;
        py_pol[i] = py_pol[i]*(1-SOR) + 
                    (-alphayx[i]*(dphi_coul[i][0]/E+4*M_PI*dphix_reac[i]) 
                     -alphayy[i]*(dphi_coul[i][1]/E+4*M_PI*dphiy_reac[i]) 
                     -alphayz[i]*(dphi_coul[i][2]/E+4*M_PI*dphiz_reac[i]))*SOR;
        pz_pol[i] = pz_pol[i]*(1-SOR) + 
                    (-alphazx[i]*(dphi_coul[i][0]/E+4*M_PI*dphix_reac[i]) 
                     -alphazy[i]*(dphi_coul[i][1]/E+4*M_PI*dphiy_reac[i]) 
                     -alphazz[i]*(dphi_coul[i][2]/E+4*M_PI*dphiz_reac[i]))*SOR;
    }
    //for (int i=0; i<10; i++) printf("%f, %f, %f\n", dphi_coul[i][0], dphi_coul[i][1], dphi_coul[i][2]);
    //printf("\n");
    
}

