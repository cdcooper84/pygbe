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

try:
    from pycuda.compiler import SourceModule
except:
    pass

def kernels(BSZ, Nm, K_fine, P, REAL):
    
    mod = SourceModule( """

    #define REAL %(precision)s
    #define BSZ %(blocksize)d
    #define Nm  %(Nmult)d
    #define K_fine %(K_near)d
    #define P      %(Ptree)d


    /*
    __device__ int getIndex(int P, int i, int j, int k)
    {
        int I=0, ii, jj; 
        for (ii=0; ii<i; ii++)
        {   
            for (jj=1; jj<P+2-ii; jj++)
            {   
                I+=jj;
            }   
        }   
        for (jj=P+2-j; jj<P+2; jj++)
        {   
            I+=jj-i;
        }   
        I+=k;

        return I;
    }
    */

    __device__ int getIndex(int i, int j, int k, int *Index)
    {   
        return Index[(P+1)*(P+1)*i + (P+1)*j + k]; 
    }

    __device__ void getCoeff(REAL *a, REAL dx, REAL dy, REAL dz, REAL kappa, int *index, int LorY)
    {
        REAL b[Nm];

        REAL R = sqrt(dx*dx+dy*dy+dz*dz);
        REAL R2 = R*R;
        REAL R3 = R2*R;
        
        int i,j,k,I,Im1x,Im2x,Im1y,Im2y,Im1z,Im2z;
        REAL C,C1,C2,Cb;

        if (LorY==2) // if Yukawa
        {   
            b[0] = exp(-kappa*R);
            a[0] = b[0]/R;
        }   

        if (LorY==1) // if Laplace
        {   
            a[0] = 1/R;
        }   

        // Two indices = 0
        I = getIndex(1,0,0, index);

        if (LorY==2) // if Yukawa
        {   
            b[I]   = -kappa * (dx*a[0]); // 1,0,0
            b[P+1] = -kappa * (dy*a[0]); // 0,1,0
            b[1]   = -kappa * (dz*a[0]); // 0,0,1

            a[I]   = -1/R2*(kappa*dx*b[0]+dx*a[0]);
            a[P+1] = -1/R2*(kappa*dy*b[0]+dy*a[0]);
            a[1]   = -1/R2*(kappa*dz*b[0]+dz*a[0]);
        
        }   

        if (LorY==1) // if Laplace
        {   
            a[I]   = -dx/R3;
            a[P+1] = -dy/R3;
            a[1]   = -dz/R3;

        }   
        
        for (i=2; i<P+1; i++)
        {   
            Cb   = -kappa/i;
            C    = 1./(i*R2);
            I    = getIndex(i,0,0, index);
            Im1x = getIndex(i-1,0,0, index);
            Im2x = getIndex(i-2,0,0, index);
            if (LorY==2) // if Yukawa
            {   
                b[I] = Cb * (dx*a[Im1x] + a[Im2x]);
                a[I] = C * ( -kappa*(dx*b[Im1x] + b[Im2x]) -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
            }

            if (LorY==1) // if Laplace
            {   
                a[I] = C * ( -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
            }

            I    = getIndex(0,i,0, index);
            Im1y = I-(P+2-i);
            Im2y = Im1y-(P+2-i+1);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dy*a[Im1y] + a[Im2y]);
                a[I] = C * ( -kappa*(dy*b[Im1y] + b[Im2y]) -(2*i-1)*dy*a[Im1y] - (i-1)*a[Im2y] );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*i-1)*dy*a[Im1y] - (i-1)*a[Im2y] );
            }

            I   = i;
            Im1z = I-1;
            Im2z = I-2;

            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dz*a[Im1z] + a[Im2z]);
                a[I] = C * ( -kappa*(dz*b[Im1z] + b[Im2z]) -(2*i-1)*dz*a[Im1z] - (i-1)*a[Im2z] );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*i-1)*dz*a[Im1z] - (i-1)*a[Im2z] );
            }
        }

        // One index = 0, one = 1 other >=1

        Cb   = -kappa/2;
        I    = getIndex(1,1,0, index);
        Im1x = P+1;
        Im1y = I-(P+2-1-1);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y]);
            a[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]) -(2*2-1)*(dx*a[Im1x]+dy*a[Im1y]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = 1./(2*R2) * ( -(2*2-1)*(dx*a[Im1x]+dy*a[Im1y]) );
        }

        I    = getIndex(1,0,1, index);
        Im1x = 1;
        Im1z = I-1;
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z]);
            a[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]) -(2*2-1)*(dx*a[Im1x]+dz*a[Im1z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = 1./(2*R2) * ( -(2*2-1)*(dx*a[Im1x]+dz*a[Im1z]) );
        }

        I    = getIndex(0,1,1, index);
        Im1y = I-(P+2-1);
        Im1z = I-1;

        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z]);
            a[I] = 1./(2*R2) * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]) -(2*2-1)*(dy*a[Im1y]+dz*a[Im1z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = 1./(2*R2) * ( -(2*2-1)*(dy*a[Im1y]+dz*a[Im1z]) );
        }

        for (i=2; i<P; i++)
        {
            Cb   = -kappa/(i+1);
            C    = 1./((1+i)*R2);
            I    = getIndex(1,i,0, index);
            Im1x = getIndex(0,i,0, index);
            Im1y = I-(P+2-i-1);
            Im2y = Im1y-(P+2-i);

            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + a[Im2y]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dx*a[Im1x]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dx*a[Im1x]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
            }

            I    = getIndex(1,0,i, index);
            Im1x = getIndex(0,0,i, index);
            Im1z = I-1;
            Im2z = I-2;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z] + a[Im2z]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dx*a[Im1x]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dx*a[Im1x]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
            }

            I    = getIndex(0,1,i, index);
            Im1y = I-(P+2-1);
            Im1z = I-1;
            Im2z = I-2;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z] + a[Im2z]);
                a[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dy*a[Im1y]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dy*a[Im1y]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
            }

            I    = getIndex(i,1,0, index);
            Im1y = I-(P+2-1-i);
            Im1x = getIndex(i-1,1,0, index);
            Im2x = getIndex(i-2,1,0, index);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dy*a[Im1y] + dx*a[Im1x] + a[Im2x]);
                a[I] = C * ( -kappa*(dy*b[Im1y]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }

            I    = getIndex(i,0,1, index);
            Im1z = I-1;
            Im1x = getIndex(i-1,0,1, index);
            Im2x = getIndex(i-2,0,1, index);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dz*a[Im1z] + dx*a[Im1x] + a[Im2x]);
                a[I] = C * ( -kappa*(dz*b[Im1z]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }

            I    = getIndex(0,i,1, index);
            Im1z = I-1;
            Im1y = I-(P+2-i);
            Im2y = Im1y-(P+2-i+1);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dz*a[Im1z] + dy*a[Im1y] + a[Im2y]);
                a[I] = C * ( -kappa*(dz*b[Im1z]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dz*a[Im1z]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dz*a[Im1z]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
            }
        }

        // One index 0, others >=2
        for (i=2; i<P+1; i++)
        {
            for (j=2; j<P+1-i; j++)
            {
                Cb   = -kappa/(i+j);
                C    = 1./((i+j)*R2);
                I    = getIndex(i,j,0, index);
                Im1x = getIndex(i-1,j,0, index);
                Im2x = getIndex(i-2,j,0, index);
                Im1y = I-(P+2-j-i);
                Im2y = Im1y-(P+3-j-i);
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + a[Im2x] + a[Im2y]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2x]+b[Im2y]) -(2*(i+j)-1)*(dx*a[Im1x]+dy*a[Im1y]) -(i+j-1)*(a[Im2x]+a[Im2y]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+j)-1)*(dx*a[Im1x]+dy*a[Im1y]) -(i+j-1)*(a[Im2x]+a[Im2y]) );
                }

                I    = getIndex(i,0,j, index);
                Im1x = getIndex(i-1,0,j, index);
                Im2x = getIndex(i-2,0,j, index);
                Im1z = I-1;
                Im2z = I-2;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z] + a[Im2x] + a[Im2z]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2x]+b[Im2z]) -(2*(i+j)-1)*(dx*a[Im1x]+dz*a[Im1z]) -(i+j-1)*(a[Im2x]+a[Im2z]) );
                }

                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+j)-1)*(dx*a[Im1x]+dz*a[Im1z]) -(i+j-1)*(a[Im2x]+a[Im2z]) );
                }

                I    = getIndex(0,i,j, index);
                Im1y = I-(P+2-i);
                Im2y = Im1y-(P+3-i);
                Im1z = I-1;
                Im2z = I-2;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z] + a[Im2y] + a[Im2z]);
                    a[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) -(2*(i+j)-1)*(dy*a[Im1y]+dz*a[Im1z]) -(i+j-1)*(a[Im2y]+a[Im2z]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+j)-1)*(dy*a[Im1y]+dz*a[Im1z]) -(i+j-1)*(a[Im2y]+a[Im2z]) );
                }
            }
        }

        if (P>2)
        {
            // Two index = 1, other>=1
            Cb   = -kappa/3;
            I    = getIndex(1,1,1, index);
            Im1x = getIndex(0,1,1, index);
            Im1y = getIndex(1,0,1, index);
            Im1y = I-(P);
            Im1z = I-1;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z]);
                a[I] = 1/(3*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]) -5*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = 1/(3*R2) * ( -5*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) );
            }

            for (i=2; i<P-1; i++)
            {
                Cb   = -kappa/(2+i);
                C    = 1./((i+2)*R2);
                I    = getIndex(i,1,1, index);
                Im1x = getIndex(i-1,1,1, index);
                Im1y = I-(P+2-i-1);
                Im1z = I-1;
                Im2x = getIndex(i-2,1,1, index);
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
                }

                I    = getIndex(1,i,1, index);
                Im1x = getIndex(0,i,1, index);
                Im1y = I-(P+2-i-1);
                Im2y = Im1y-(P+3-i-1);
                Im1z = I-1 ;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2y]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2y]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2y]) );
                }


                I    = getIndex(1,1,i, index);
                Im1x = getIndex(0,1,i, index);
                Im1y = I-(P);
                Im1z = I-1;
                Im2z = I-2;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2z]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2z]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2z]) );
                }
            }
        }

        // One index = 1, others >=2
        if (P>4)
        {
            for (i=2; i<P-2; i++)
            {
                for (j=2; j<P-i; j++)
                {
                    Cb = -kappa/(1+i+j);
                    C  = 1./((1+i+j)*R2);
                    C1 = -(2.*(1+i+j)-1);
                    C2 = (i+j);
                    I    = getIndex(1,i,j, index);
                    Im1x = getIndex(0,i,j, index);
                    Im1y = I-(P+2-1-i);
                    Im2y = Im1y-(P+3-1-i);
                    Im1z = I-1;
                    Im2z = I-2;
                    if (LorY==2) // if Yukawa
                    {
                        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2y] + a[Im2z]);
                        a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2y]+a[Im2z]) );
                    }
                    if (LorY==1) // if Laplace
                    {
                        a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2y]+a[Im2z]) );
                    }

                    I    = getIndex(i,1,j, index);
                    Im1x = getIndex(i-1,1,j, index);
                    Im1y = I-(P+2-i-1);
                    Im2x = getIndex(i-2,1,j, index);
                    Im1z = I-1;
                    Im2z = I-2;
                    if (LorY==2) // if Yukawa
                    {
                        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2z]);
                        a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2z]) );
                    }
                    if (LorY==1) // if Laplace
                    {
                        a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2z]) );
                    }

                    I    = getIndex(i,j,1, index);
                    Im1x = getIndex(i-1,j,1, index);
                    Im2x = getIndex(i-2,j,1, index);
                    Im1y = I-(P+2-i-j);
                    Im2y = Im1y-(P+3-i-j);
                    Im1z = I-1;
                    if (LorY==2) // if Yukawa
                    {
                        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2y]);
                        a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]) );
                    }
                    if (LorY==1) // if Laplace
                    {
                        a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]) );
                    }
                }
            }
        }

        // All indices >= 2
        if (P>5)
        {
            for (i=2;i<P-3;i++)
            {
                for (j=2;j<P-1-i;j++)
                {
                    for (k=2;k<P+1-i-j;k++)
                    {
                        Cb = -kappa/(i+j+k);
                        C  = 1./((i+j+k)*R2);
                        C1 = -(2.*(i+j+k)-1);
                        C2 = i+j+k-1.;
                        I    = getIndex(i,j,k, index);
                        Im1x = getIndex(i-1,j,k, index);
                        Im2x = getIndex(i-2,j,k, index);
                        Im1y = I-(P+2-i-j);
                        Im2y = Im1y-(P+3-i-j);
                        Im1z = I-1;
                        Im2z = I-2;
                        if (LorY==2) // if Yukawa
                        {
                            b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2y] + a[Im2z]);
                            a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]+a[Im2z]) );
                        }

                        if (LorY==1) // if Laplace
                        {
                            a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]+a[Im2z]) );
                        }
                    }
                }
            }
        }
    }

    __device__ void getCoeff_shift(REAL *ax, REAL *ay, REAL *az, REAL dx, REAL dy, REAL dz, REAL kappa, int *index, int LorY)
    {
        REAL b[Nm], a[Nm];

        REAL R = sqrt(dx*dx+dy*dy+dz*dz);
        REAL R2 = R*R;
        REAL R3 = R2*R;
        
        int i,j,k,I,Im1x,Im2x,Im1y,Im2y,Im1z,Im2z;
        REAL C,C1,C2,Cb;

        if (LorY==2) // if Yukawa
        {   
            b[0] = exp(-kappa*R);
            a[0] = b[0]/R;
        }   

        if (LorY==1) // if Laplace
        {   
            a[0] = 1/R;
        }   

        // Two indices = 0
        I = getIndex(1,0,0, index);

        if (LorY==2) // if Yukawa
        {   
            b[I]   = -kappa * (dx*a[0]); // 1,0,0
            b[P+1] = -kappa * (dy*a[0]); // 0,1,0
            b[1]   = -kappa * (dz*a[0]); // 0,0,1

            a[I]   = -1/R2*(kappa*dx*b[0]+dx*a[0]);
            a[P+1] = -1/R2*(kappa*dy*b[0]+dy*a[0]);
            a[1]   = -1/R2*(kappa*dz*b[0]+dz*a[0]);
        
        }   

        if (LorY==1) // if Laplace
        {   
            a[I]   = -dx/R3;
            a[P+1] = -dy/R3;
            a[1]   = -dz/R3;

        }   
        
        ax[0] = a[I];
        ay[0] = a[P+1];
        az[0] = a[1];

        for (i=2; i<P+1; i++)
        {   
            Cb   = -kappa/i;
            C    = 1./(i*R2);
            I    = getIndex(i,0,0, index);
            Im1x = getIndex(i-1,0,0, index);
            Im2x = getIndex(i-2,0,0, index);
            if (LorY==2) // if Yukawa
            {   
                b[I] = Cb * (dx*a[Im1x] + a[Im2x]);
                a[I] = C * ( -kappa*(dx*b[Im1x] + b[Im2x]) -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
            }

            if (LorY==1) // if Laplace
            {   
                a[I] = C * ( -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
            }

            ax[Im1x] = a[I]*i;

            I    = getIndex(0,i,0, index);
            Im1y = I-(P+2-i);
            Im2y = Im1y-(P+2-i+1);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dy*a[Im1y] + a[Im2y]);
                a[I] = C * ( -kappa*(dy*b[Im1y] + b[Im2y]) -(2*i-1)*dy*a[Im1y] - (i-1)*a[Im2y] );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*i-1)*dy*a[Im1y] - (i-1)*a[Im2y] );
            }

            ay[Im1y] = a[I]*i;

            I   = i;
            Im1z = I-1;
            Im2z = I-2;

            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dz*a[Im1z] + a[Im2z]);
                a[I] = C * ( -kappa*(dz*b[Im1z] + b[Im2z]) -(2*i-1)*dz*a[Im1z] - (i-1)*a[Im2z] );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*i-1)*dz*a[Im1z] - (i-1)*a[Im2z] );
            }

            az[Im1z] = a[I]*i;
        }

        // One index = 0, one = 1 other >=1

        Cb   = -kappa/2;
        I    = getIndex(1,1,0, index);
        Im1x = P+1;
        Im1y = I-(P+2-1-1);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y]);
            a[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]) -(2*2-1)*(dx*a[Im1x]+dy*a[Im1y]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = 1./(2*R2) * ( -(2*2-1)*(dx*a[Im1x]+dy*a[Im1y]) );
        }

        ax[Im1x] = a[I];
        ay[Im1y] = a[I];

        I    = getIndex(1,0,1, index);
        Im1x = 1;
        Im1z = I-1;
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z]);
            a[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]) -(2*2-1)*(dx*a[Im1x]+dz*a[Im1z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = 1./(2*R2) * ( -(2*2-1)*(dx*a[Im1x]+dz*a[Im1z]) );
        }

        ax[Im1x] = a[I];
        az[Im1z] = a[I];

        I    = getIndex(0,1,1, index);
        Im1y = I-(P+2-1);
        Im1z = I-1;

        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z]);
            a[I] = 1./(2*R2) * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]) -(2*2-1)*(dy*a[Im1y]+dz*a[Im1z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = 1./(2*R2) * ( -(2*2-1)*(dy*a[Im1y]+dz*a[Im1z]) );
        }

        ay[Im1y] = a[I];
        az[Im1z] = a[I];

        for (i=2; i<P; i++)
        {
            Cb   = -kappa/(i+1);
            C    = 1./((1+i)*R2);
            I    = getIndex(1,i,0, index);
            Im1x = getIndex(0,i,0, index);
            Im1y = I-(P+2-i-1);
            Im2y = Im1y-(P+2-i);

            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + a[Im2y]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dx*a[Im1x]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dx*a[Im1x]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
            }

            ax[Im1x] = a[I];
            ay[Im1y] = a[I];

            I    = getIndex(1,0,i, index);
            Im1x = getIndex(0,0,i, index);
            Im1z = I-1;
            Im2z = I-2;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z] + a[Im2z]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dx*a[Im1x]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dx*a[Im1x]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
            }

            ax[Im1x] = a[I];
            az[Im1z] = a[I]*i;

            I    = getIndex(0,1,i, index);
            Im1y = I-(P+2-1);
            Im1z = I-1;
            Im2z = I-2;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z] + a[Im2z]);
                a[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dy*a[Im1y]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dy*a[Im1y]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
            }

            ay[Im1y] = a[I];
            az[Im1z] = a[I]*i;

            I    = getIndex(i,1,0, index);
            Im1y = I-(P+2-1-i);
            Im1x = getIndex(i-1,1,0, index);
            Im2x = getIndex(i-2,1,0, index);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dy*a[Im1y] + dx*a[Im1x] + a[Im2x]);
                a[I] = C * ( -kappa*(dy*b[Im1y]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }

            ay[Im1y] = a[I];
            ax[Im1x] = a[I]*i;

            I    = getIndex(i,0,1, index);
            Im1z = I-1;
            Im1x = getIndex(i-1,0,1, index);
            Im2x = getIndex(i-2,0,1, index);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dz*a[Im1z] + dx*a[Im1x] + a[Im2x]);
                a[I] = C * ( -kappa*(dz*b[Im1z]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }

            az[Im1z] = a[I];
            ax[Im1x] = a[I]*i;

            I    = getIndex(0,i,1, index);
            Im1z = I-1;
            Im1y = I-(P+2-i);
            Im2y = Im1y-(P+2-i+1);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dz*a[Im1z] + dy*a[Im1y] + a[Im2y]);
                a[I] = C * ( -kappa*(dz*b[Im1z]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dz*a[Im1z]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dz*a[Im1z]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
            }

            az[Im1z] = a[I];
            ay[Im1y] = a[I]*i;

        }

        // One index 0, others >=2
        for (i=2; i<P+1; i++)
        {
            for (j=2; j<P+1-i; j++)
            {
                Cb   = -kappa/(i+j);
                C    = 1./((i+j)*R2);
                I    = getIndex(i,j,0, index);
                Im1x = getIndex(i-1,j,0, index);
                Im2x = getIndex(i-2,j,0, index);
                Im1y = I-(P+2-j-i);
                Im2y = Im1y-(P+3-j-i);
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + a[Im2x] + a[Im2y]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2x]+b[Im2y]) -(2*(i+j)-1)*(dx*a[Im1x]+dy*a[Im1y]) -(i+j-1)*(a[Im2x]+a[Im2y]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+j)-1)*(dx*a[Im1x]+dy*a[Im1y]) -(i+j-1)*(a[Im2x]+a[Im2y]) );
                }

                ax[Im1x] = a[I]*i;
                ay[Im1y] = a[I]*j;

                I    = getIndex(i,0,j, index);
                Im1x = getIndex(i-1,0,j, index);
                Im2x = getIndex(i-2,0,j, index);
                Im1z = I-1;
                Im2z = I-2;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z] + a[Im2x] + a[Im2z]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2x]+b[Im2z]) -(2*(i+j)-1)*(dx*a[Im1x]+dz*a[Im1z]) -(i+j-1)*(a[Im2x]+a[Im2z]) );
                }

                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+j)-1)*(dx*a[Im1x]+dz*a[Im1z]) -(i+j-1)*(a[Im2x]+a[Im2z]) );
                }

                ax[Im1x] = a[I]*i;
                az[Im1z] = a[I]*j;

                I    = getIndex(0,i,j, index);
                Im1y = I-(P+2-i);
                Im2y = Im1y-(P+3-i);
                Im1z = I-1;
                Im2z = I-2;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z] + a[Im2y] + a[Im2z]);
                    a[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) -(2*(i+j)-1)*(dy*a[Im1y]+dz*a[Im1z]) -(i+j-1)*(a[Im2y]+a[Im2z]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+j)-1)*(dy*a[Im1y]+dz*a[Im1z]) -(i+j-1)*(a[Im2y]+a[Im2z]) );
                }

                ay[Im1y] = a[I]*i;
                az[Im1z] = a[I]*j;

            }
        }

        if (P>2)
        {
            // Two index = 1, other>=1
            Cb   = -kappa/3;
            I    = getIndex(1,1,1, index);
            Im1x = getIndex(0,1,1, index);
            Im1y = getIndex(1,0,1, index);
            Im1y = I-(P);
            Im1z = I-1;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z]);
                a[I] = 1/(3*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]) -5*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = 1/(3*R2) * ( -5*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) );
            }

            ax[Im1x] = a[I];
            ay[Im1y] = a[I];
            az[Im1z] = a[I];

            for (i=2; i<P-1; i++)
            {
                Cb   = -kappa/(2+i);
                C    = 1./((i+2)*R2);
                I    = getIndex(i,1,1, index);
                Im1x = getIndex(i-1,1,1, index);
                Im1y = I-(P+2-i-1);
                Im1z = I-1;
                Im2x = getIndex(i-2,1,1, index);
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
                }

                ax[Im1x] = a[I]*i;
                ay[Im1y] = a[I];
                az[Im1z] = a[I];

                I    = getIndex(1,i,1, index);
                Im1x = getIndex(0,i,1, index);
                Im1y = I-(P+2-i-1);
                Im2y = Im1y-(P+3-i-1);
                Im1z = I-1 ;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2y]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2y]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2y]) );
                }

                ax[Im1x] = a[I];
                ay[Im1y] = a[I]*i;
                az[Im1z] = a[I];

                I    = getIndex(1,1,i, index);
                Im1x = getIndex(0,1,i, index);
                Im1y = I-(P);
                Im1z = I-1;
                Im2z = I-2;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2z]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2z]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2z]) );
                }

                ax[Im1x] = a[I];
                ay[Im1y] = a[I];
                az[Im1z] = a[I]*i;

            }
        }

        // One index = 1, others >=2
        if (P>4)
        {
            for (i=2; i<P-2; i++)
            {
                for (j=2; j<P-i; j++)
                {
                    Cb = -kappa/(1+i+j);
                    C  = 1./((1+i+j)*R2);
                    C1 = -(2.*(1+i+j)-1);
                    C2 = (i+j);
                    I    = getIndex(1,i,j, index);
                    Im1x = getIndex(0,i,j, index);
                    Im1y = I-(P+2-1-i);
                    Im2y = Im1y-(P+3-1-i);
                    Im1z = I-1;
                    Im2z = I-2;
                    if (LorY==2) // if Yukawa
                    {
                        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2y] + a[Im2z]);
                        a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2y]+a[Im2z]) );
                    }
                    if (LorY==1) // if Laplace
                    {
                        a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2y]+a[Im2z]) );
                    }

                    ax[Im1x] = a[I];
                    ay[Im1y] = a[I]*i;
                    az[Im1z] = a[I]*j;

                    I    = getIndex(i,1,j, index);
                    Im1x = getIndex(i-1,1,j, index);
                    Im1y = I-(P+2-i-1);
                    Im2x = getIndex(i-2,1,j, index);
                    Im1z = I-1;
                    Im2z = I-2;
                    if (LorY==2) // if Yukawa
                    {
                        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2z]);
                        a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2z]) );
                    }
                    if (LorY==1) // if Laplace
                    {
                        a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2z]) );
                    }

                    ax[Im1x] = a[I]*i;
                    ay[Im1y] = a[I];
                    az[Im1z] = a[I]*j;

                    I    = getIndex(i,j,1, index);
                    Im1x = getIndex(i-1,j,1, index);
                    Im2x = getIndex(i-2,j,1, index);
                    Im1y = I-(P+2-i-j);
                    Im2y = Im1y-(P+3-i-j);
                    Im1z = I-1;
                    if (LorY==2) // if Yukawa
                    {
                        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2y]);
                        a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]) );
                    }
                    if (LorY==1) // if Laplace
                    {
                        a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]) );
                    }

                    ax[Im1x] = a[I]*i;
                    ay[Im1y] = a[I]*j;
                    az[Im1z] = a[I];

                }
            }
        }

        // All indices >= 2
        if (P>5)
        {
            for (i=2;i<P-3;i++)
            {
                for (j=2;j<P-1-i;j++)
                {
                    for (k=2;k<P+1-i-j;k++)
                    {
                        Cb = -kappa/(i+j+k);
                        C  = 1./((i+j+k)*R2);
                        C1 = -(2.*(i+j+k)-1);
                        C2 = i+j+k-1.;
                        I    = getIndex(i,j,k, index);
                        Im1x = getIndex(i-1,j,k, index);
                        Im2x = getIndex(i-2,j,k, index);
                        Im1y = I-(P+2-i-j);
                        Im2y = Im1y-(P+3-i-j);
                        Im1z = I-1;
                        Im2z = I-2;
                        if (LorY==2) // if Yukawa
                        {
                            b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2y] + a[Im2z]);
                            a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]+a[Im2z]) );
                        }

                        if (LorY==1) // if Laplace
                        {
                            a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]+a[Im2z]) );
                        }

                        ax[Im1x] = a[I]*i;
                        ay[Im1y] = a[I]*j;
                        az[Im1z] = a[I]*k;

                    }
                }
            }
        }
    }

    __device__ void multipole(REAL &K, REAL &V, REAL *M, REAL *Md,
                            REAL *a, int CJ_start, int jblock, int j)
    {
        int offset;
        for (int i=0; i<Nm; i++)
        {
            offset = (CJ_start+j)*Nm + jblock*BSZ*Nm + i;
            V += M[offset] * a[i];
            K += Md[offset]* a[i]; 
        }

    }

    __device__ void multipoleKt(REAL &Ktx, REAL &Kty, REAL &Ktz, REAL *M,
                            REAL *ax, REAL *ay, REAL *az, int CJ_start, int jblock, int j)
    {
        int offset;
        for (int i=0; i<Nm; i++)
        {
            offset = (CJ_start+j)*Nm + jblock*BSZ*Nm + i;
            Ktx += M[offset] * ax[i];
            Kty += M[offset] * ay[i];
            Ktz += M[offset] * az[i];
        }

    }

    __device__ REAL mynorm(REAL *x)
    {
        return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    }

    __device__ void cross(REAL *x, REAL *y, REAL *z) // z is the resulting array
    {
        z[0] = x[1]*y[2] - x[2]*y[1];
        z[1] = x[2]*y[0] - x[0]*y[2];
        z[2] = x[0]*y[1] - x[1]*y[0];
    }

    __device__ void MV(REAL *M, REAL *V, REAL *res) // 3x3 mat-vec
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

    __device__ void MVip(REAL *M, REAL *V) // 3x3 mat-vec in-place
    {
        REAL V2[3] = {V[0], V[1], V[2]};
        for (int i=0; i<3; i++)
        {
            REAL sum = 0.;
            for (int j=0; j<3; j++)
            {
                sum += M[3*i+j]*V2[j];
            }
            V[i] = sum;
        }
    }

    __device__ REAL dot_prod(REAL *x, REAL *y) // len(3) vector dot product
    {
        return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
    }

    __device__ void axpy(REAL *x, REAL *y, REAL *z, REAL alpha, int sign, int N)
    {
        for(int i=0; i<N; i++)
        {
            z[i] = sign*alpha*x[i] + y[i];
        }
    }

    __device__ void ax(REAL *x, REAL *y, REAL alpha, int N)
    {
        for(int i=0; i<N; i++)
        {
            y[i] = alpha*x[i];
        }

    }

    __device__ void axip(REAL *x, REAL alpha, int N)
    {
        for(int i=0; i<N; i++)
        {
            x[i] = alpha*x[i];
        }

    }

    __device__ void lineInt(REAL &PHI_K, REAL &PHI_V, REAL z, REAL x, REAL v1, REAL v2, REAL kappa, REAL *xk, REAL *wk, int K, int LorY)
    {
        REAL theta1 = atan2(v1,x);
        REAL theta2 = atan2(v2,x);

        REAL absZ = fabs(z), signZ;
        if (absZ<1e-10) signZ = 0;
        else            signZ = z/absZ;

        // Loop over gauss points
        REAL thetak, Rtheta, R, expKr, expKz = exp(-kappa*absZ);
        for (int i=0; i<K; i++)
        {
            thetak = (theta2 - theta1)/2*xk[i] + (theta2 + theta1)/2;
            Rtheta = x/cos(thetak);
            R      = sqrt(Rtheta*Rtheta + z*z);
            expKr  = exp(-kappa*R);
            if (LorY==2)
            {
                if (kappa>1e-12)
                {
                    PHI_V+= -wk[i]*(expKr - expKz)/kappa * (theta2 - theta1)/2;
                    PHI_K+=  wk[i]*(z/R*expKr - expKz*signZ) * (theta2 - theta1)/2;
                }
                else
                {
                    PHI_V+= wk[i]*(R-absZ) * (theta2 - theta1)/2;
                    PHI_K+= wk[i]*(z/R - signZ) * (theta2 - theta1)/2;
                }
            }

            if (LorY==1)
            {
                PHI_V += wk[i]*(R-absZ) * (theta2 - theta1)/2;
                PHI_K += wk[i]*(z/R - signZ) * (theta2 - theta1)/2;
            }
        }
    }

    __device__ void intSide(REAL &PHI_K, REAL &PHI_V, REAL *v1, REAL *v2, REAL p, REAL kappa, REAL *xk, REAL *wk, int K, int LorY)
    {
        REAL v21u[3];
        for (int i=0; i<3; i++)
        {
            v21u[i] = v2[i] - v1[i];
        }

        REAL L21 = mynorm(v21u);
        axip(v21u, 1/L21, 3);

        REAL unit[3] = {0.,0.,1.};
        REAL orthog[3];
        cross(unit, v21u, orthog);

        REAL v1new_x = dot_prod(orthog, v1); 
        REAL v1new_y = dot_prod(v21u, v1); 

        if (v1new_x<0)
        {
            axip(v21u, -1, 3);
            axip(orthog, -1, 3);
            v1new_x = dot_prod(orthog, v1);
            v1new_y = dot_prod(v21u, v1);
        }

        REAL v2new_y = dot_prod(v21u, v2); 

        if ((v1new_y>0 && v2new_y<0) || (v1new_y<0 && v2new_y>0))
        {
            lineInt(PHI_K, PHI_V, p, v1new_x, 0, v1new_y, kappa, xk, wk, K, LorY);
            lineInt(PHI_K, PHI_V, p, v1new_x, v2new_y, 0, kappa, xk, wk, K, LorY);

        }
        else
        {
            REAL PHI_Kaux = 0., PHI_Vaux = 0.;
            lineInt(PHI_Kaux, PHI_Vaux, p, v1new_x, v1new_y, v2new_y, kappa, xk, wk, K, LorY);

            PHI_K -= PHI_Kaux;
            PHI_V -= PHI_Vaux;
        }
    }

    __device__ void SA(REAL &PHI_K, REAL &PHI_V, REAL *y, REAL x0, REAL x1, REAL x2, 
                       REAL K_diag, REAL V_diag, REAL kappa, int same, REAL *xk, REAL *wk, int K, int LorY)
    {   
        REAL y0_panel[3], y1_panel[3], y2_panel[3], x_panel[3];
        REAL X[3], Y[3], Z[3];

        x_panel[0] = x0 - y[0];
        x_panel[1] = x1 - y[1];
        x_panel[2] = x2 - y[2];
        for (int i=0; i<3; i++)
        {
            y0_panel[i] = 0.;
            y1_panel[i] = y[3+i] - y[i];
            y2_panel[i] = y[6+i] - y[i];
            X[i] = y1_panel[i];
        }


        // Find panel coordinate system X: 0->1
        cross(y1_panel, y2_panel, Z); 
        REAL Xnorm = mynorm(X); 
        REAL Znorm = mynorm(Z); 
        for (int i=0; i<3; i++)
        {   
            X[i] /= Xnorm;
            Z[i] /= Znorm;
        }   

        cross(Z,X,Y);

        // Rotate the coordinate system to match panel plane
        // Multiply y_panel times a rotation matrix [X; Y; Z]
        REAL x_aux, y_aux, z_aux;
        x_aux = dot_prod(X, y0_panel);
        y_aux = dot_prod(Y, y0_panel);
        z_aux = dot_prod(Z, y0_panel);
        y0_panel[0] = x_aux;
        y0_panel[1] = y_aux;
        y0_panel[2] = z_aux;

        x_aux = dot_prod(X, y1_panel);
        y_aux = dot_prod(Y, y1_panel);
        z_aux = dot_prod(Z, y1_panel);
        y1_panel[0] = x_aux;
        y1_panel[1] = y_aux;
        y1_panel[2] = z_aux;

        x_aux = dot_prod(X, y2_panel);
        y_aux = dot_prod(Y, y2_panel);
        z_aux = dot_prod(Z, y2_panel);
        y2_panel[0] = x_aux;
        y2_panel[1] = y_aux;
        y2_panel[2] = z_aux;

        x_aux = dot_prod(X, x_panel);
        y_aux = dot_prod(Y, x_panel);
        z_aux = dot_prod(Z, x_panel);
        x_panel[0] = x_aux;
        x_panel[1] = y_aux;
        x_panel[2] = z_aux;

        // Shift origin so it matches collocation point
        for (int i=0; i<2; i++)
        {   
            y0_panel[i] -= x_panel[i]; 
            y1_panel[i] -= x_panel[i]; 
            y2_panel[i] -= x_panel[i]; 
        }   

        // Loop over sides
        intSide(PHI_K, PHI_V, y0_panel, y1_panel, x_panel[2], kappa, xk, wk, K, LorY); // Side 0
        intSide(PHI_K, PHI_V, y1_panel, y2_panel, x_panel[2], kappa, xk, wk, K, LorY); // Side 1
        intSide(PHI_K, PHI_V, y2_panel, y0_panel, x_panel[2], kappa, xk, wk, K, LorY); // Side 2

        if (same==1)
        {
            PHI_K += K_diag;
            PHI_V += V_diag;
        }
        
    }

    __device__ __inline__ void GQ_fine(REAL &PHI_K, REAL &PHI_V, REAL *panel, int J, REAL xi, REAL yi, REAL zi, 
                            REAL kappa, REAL *Xk, REAL *Wk, REAL *Area, int LorY)
    {
        REAL nx, ny, nz;
        REAL dx, dy, dz, r, aux;

        PHI_K = 0.;
        PHI_V = 0.;
        int j = J/9;

        aux = 1/(2*Area[j]);
        nx = ((panel[J+4]-panel[J+1])*(panel[J+2]-panel[J+8]) - (panel[J+5]-panel[J+2])*(panel[J+1]-panel[J+7])) * aux;
        ny = ((panel[J+5]-panel[J+2])*(panel[J+0]-panel[J+6]) - (panel[J+3]-panel[J+0])*(panel[J+2]-panel[J+8])) * aux;
        nz = ((panel[J+3]-panel[J+0])*(panel[J+1]-panel[J+7]) - (panel[J+4]-panel[J+1])*(panel[J+0]-panel[J+6])) * aux;

        #pragma unroll
        for (int kk=0; kk<K_fine; kk++)
        {
            dx = xi - (panel[J+0]*Xk[3*kk] + panel[J+3]*Xk[3*kk+1] + panel[J+6]*Xk[3*kk+2]);
            dy = yi - (panel[J+1]*Xk[3*kk] + panel[J+4]*Xk[3*kk+1] + panel[J+7]*Xk[3*kk+2]);
            dz = zi - (panel[J+2]*Xk[3*kk] + panel[J+5]*Xk[3*kk+1] + panel[J+8]*Xk[3*kk+2]);
            r   = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!

            if (LorY==1)
            {
                aux = Wk[kk]*Area[j]*r;
                PHI_V += aux;
                PHI_K += aux*(nx*dx+ny*dy+nz*dz)*(r*r);
            }

            else
            {
                aux = Wk[kk]*Area[j]*exp(-kappa*1/r)*r;
                PHI_V += aux;
                PHI_K += aux*(nx*dx+ny*dy+nz*dz)*r*(kappa+r);
            }

        }
    }

    __device__ __inline__ void GQ_fine_derivative(REAL &dPHI_Kx, REAL &dPHI_Vx, 
                            REAL &dPHI_Ky, REAL &dPHI_Vy, REAL &dPHI_Kz, REAL &dPHI_Vz, 
                            REAL *panel, int J, REAL xi, REAL yi, REAL zi, 
                            REAL kappa, REAL *Xk, REAL *Wk, REAL *Area, int LorY)
    {
        REAL nx, ny, nz;
        REAL dx, dy, dz, r, r3, aux;

        dPHI_Kx = 0.;
        dPHI_Vx = 0.;
        dPHI_Ky = 0.;
        dPHI_Vy = 0.;
        dPHI_Kz = 0.;
        dPHI_Vz = 0.;
        int j = J/9;

        aux = 1/(2*Area[j]);
        nx = ((panel[J+4]-panel[J+1])*(panel[J+2]-panel[J+8]) - (panel[J+5]-panel[J+2])*(panel[J+1]-panel[J+7])) * aux;
        ny = ((panel[J+5]-panel[J+2])*(panel[J+0]-panel[J+6]) - (panel[J+3]-panel[J+0])*(panel[J+2]-panel[J+8])) * aux;
        nz = ((panel[J+3]-panel[J+0])*(panel[J+1]-panel[J+7]) - (panel[J+4]-panel[J+1])*(panel[J+0]-panel[J+6])) * aux;

        #pragma unroll
        for (int kk=0; kk<K_fine; kk++)
        {
            dx = xi - (panel[J+0]*Xk[3*kk] + panel[J+3]*Xk[3*kk+1] + panel[J+6]*Xk[3*kk+2]);
            dy = yi - (panel[J+1]*Xk[3*kk] + panel[J+4]*Xk[3*kk+1] + panel[J+7]*Xk[3*kk+2]);
            dz = zi - (panel[J+2]*Xk[3*kk] + panel[J+5]*Xk[3*kk+1] + panel[J+8]*Xk[3*kk+2]);
            r  = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!
            r3 = r*r*r;

            if (LorY==1)
            {
                aux = Wk[kk]*Area[j]*r3;
                dPHI_Vx -= dx*aux;
                dPHI_Vy -= dy*aux;
                dPHI_Vz -= dz*aux;
                dPHI_Kx += aux*nx-3*aux*dx*(nx*dx+ny*dy+nz*dz)*(r*r);
                dPHI_Ky += aux*ny-3*aux*dy*(nx*dx+ny*dy+nz*dz)*(r*r);
                dPHI_Kz += aux*nz-3*aux*dz*(nx*dx+ny*dy+nz*dz)*(r*r);
            }

            else // this will never fire because it is always laplace in this case
            {
                aux = Wk[kk]*Area[j]*exp(-kappa*1/r)*r;
                dPHI_Vx += aux;
                dPHI_Kx += aux*(nx*dx+ny*dy+nz*dz)*r*(kappa+r);
            }

        }
    }

    __device__ __inline__ void GQ_fine_2derivative(REAL &ddPHI_Kxx, REAL &ddPHI_Vxx, 
                            REAL &ddPHI_Kxy, REAL &ddPHI_Vxy, REAL &ddPHI_Kxz, REAL &ddPHI_Vxz, 
                            REAL &ddPHI_Kyx, REAL &ddPHI_Vyx, REAL &ddPHI_Kyy, REAL &ddPHI_Vyy, REAL &ddPHI_Kyz, REAL &ddPHI_Vyz,
                            REAL &ddPHI_Kzx, REAL &ddPHI_Vzx, REAL &ddPHI_Kzy, REAL &ddPHI_Vzy, REAL &ddPHI_Kzz, REAL &ddPHI_Vzz,
                            REAL *panel, int J, REAL xi, REAL yi, REAL zi, 
                            REAL kappa, REAL *Xk, REAL *Wk, REAL *Area, int LorY)
    {
        REAL nx, ny, nz;
        REAL dx, dy, dz, r, r2, r3, aux;

        ddPHI_Kxx = 0.;
        ddPHI_Vxx = 0.;
        ddPHI_Kxy = 0.;
        ddPHI_Vxy = 0.;
        ddPHI_Kxz = 0.;
        ddPHI_Vxz = 0.;
        ddPHI_Kyx = 0.;
        ddPHI_Vyx = 0.;
        ddPHI_Kyy = 0.;
        ddPHI_Vyy = 0.;
        ddPHI_Kyz = 0.;
        ddPHI_Vyz = 0.;
        ddPHI_Kzx = 0.;
        ddPHI_Vzx = 0.;
        ddPHI_Kzy = 0.;
        ddPHI_Vzy = 0.;
        ddPHI_Kzz = 0.;
        ddPHI_Vzz = 0.;
        int j = J/9;

        aux = 1/(2*Area[j]);
        nx = ((panel[J+4]-panel[J+1])*(panel[J+2]-panel[J+8]) - (panel[J+5]-panel[J+2])*(panel[J+1]-panel[J+7])) * aux;
        ny = ((panel[J+5]-panel[J+2])*(panel[J+0]-panel[J+6]) - (panel[J+3]-panel[J+0])*(panel[J+2]-panel[J+8])) * aux;
        nz = ((panel[J+3]-panel[J+0])*(panel[J+1]-panel[J+7]) - (panel[J+4]-panel[J+1])*(panel[J+0]-panel[J+6])) * aux;

        #pragma unroll
        for (int kk=0; kk<K_fine; kk++)
        {
            dx = xi - (panel[J+0]*Xk[3*kk] + panel[J+3]*Xk[3*kk+1] + panel[J+6]*Xk[3*kk+2]);
            dy = yi - (panel[J+1]*Xk[3*kk] + panel[J+4]*Xk[3*kk+1] + panel[J+7]*Xk[3*kk+2]);
            dz = zi - (panel[J+2]*Xk[3*kk] + panel[J+5]*Xk[3*kk+1] + panel[J+8]*Xk[3*kk+2]);
            r  = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!
            r2 = r*r;
            r3 = r2*r;

            if (LorY==1)
            {
                aux = Wk[kk]*Area[j]*r3;
                ddPHI_Vxx += aux*(-1+3*dx*dx*r2);
                ddPHI_Vxy += aux*3*dx*dy*r2;
                ddPHI_Vxz += aux*3*dx*dz*r2;
                ddPHI_Vyx += aux*3*dy*dx*r2;
                ddPHI_Vyy += aux*(-1+3*dy*dy*r2);
                ddPHI_Vxz += aux*3*dy*dz*r2;
                ddPHI_Vzx += aux*3*dz*dx*r2;
                ddPHI_Vzy += aux*3*dz*dy*r2;
                ddPHI_Vzz += aux*(-1+3*dz*dz*r2);
                ddPHI_Kxx += -3*aux*r2*(3*dx*nx + dy*ny + dz*nz - 5*r2*dx*dx*(dx*nx+dy*ny+dz*nz)); 
                ddPHI_Kxy += -3*aux*r2*(dx*ny + dy*nx - 5*r2*dx*dy*(dx*nx+dy*ny+dz*nz)); 
                ddPHI_Kxz += -3*aux*r2*(dx*nz + dz*nx - 5*r2*dx*dz*(dx*nx+dy*ny+dz*nz)); 
                ddPHI_Kyx += -3*aux*r2*(dy*nx + dx*ny - 5*r2*dy*dx*(dx*nx+dy*ny+dz*nz)); 
                ddPHI_Kyy += -3*aux*r2*(3*dy*ny + dx*nx + dz*nz - 5*r2*dy*dy*(dx*nx+dy*ny+dz*nz)); 
                ddPHI_Kyz += -3*aux*r2*(dy*nz + dz*ny - 5*r2*dy*dz*(dx*nx+dy*ny+dz*nz)); 
                ddPHI_Kzx += -3*aux*r2*(dz*nx + dx*nz - 5*r2*dz*dx*(dx*nx+dy*ny+dz*nz)); 
                ddPHI_Kzy += -3*aux*r2*(dz*ny + dy*nz - 5*r2*dz*dy*(dx*nx+dy*ny+dz*nz)); 
                ddPHI_Kzz += -3*aux*r2*(3*dz*nz + dy*ny + dx*nx - 5*r2*dz*dz*(dx*nx+dy*ny+dz*nz)); 

            }

            else // this will never fire because it is always laplace in this case
            {
                aux = Wk[kk]*Area[j]*exp(-kappa*1/r)*r;
                ddPHI_Vxx += aux;
                ddPHI_Kxx += aux*(nx*dx+ny*dy+nz*dz)*r*(kappa+r);
            }

        }
    }

    __device__ __inline__ void GQ_fineKt(REAL &PHI_Ktx, REAL &PHI_Kty, REAL &PHI_Ktz, REAL *panel, int J, 
                            REAL xi, REAL yi, REAL zi, REAL kappa, REAL *Xk, REAL *Wk, REAL *Area, int LorY)
    {
        REAL dx, dy, dz, r, aux;

        PHI_Ktx = 0.;
        PHI_Kty = 0.;
        PHI_Ktz = 0.;
        int j = J/9;

        #pragma unroll
        for (int kk=0; kk<K_fine; kk++)
        {
            dx = xi - (panel[J+0]*Xk[3*kk] + panel[J+3]*Xk[3*kk+1] + panel[J+6]*Xk[3*kk+2]);
            dy = yi - (panel[J+1]*Xk[3*kk] + panel[J+4]*Xk[3*kk+1] + panel[J+7]*Xk[3*kk+2]);
            dz = zi - (panel[J+2]*Xk[3*kk] + panel[J+5]*Xk[3*kk+1] + panel[J+8]*Xk[3*kk+2]);
            r   = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!

            if (LorY==1)
            {
                aux = Wk[kk]*Area[j]*r*r*r;
                PHI_Ktx -= aux*dx;
                PHI_Kty -= aux*dy;
                PHI_Ktz -= aux*dz;
            }

            else
            {
                aux = Wk[kk]*Area[j]*exp(-kappa*1/r)*r*r*(kappa+r);
                PHI_Ktx -= aux*dx;
                PHI_Kty -= aux*dy;
                PHI_Ktz -= aux*dz;
            }
        }
    }


    __global__ void M2P(REAL *K_gpu, REAL *V_gpu, int *offMlt, int *sizeTar, REAL *xc, REAL *yc, REAL *zc, 
                        REAL *M, REAL *Md, REAL *xt, REAL *yt, REAL *zt,
                        int *Index, int ptr_off, int ptr_lst, REAL kappa, int BpT, int NCRIT, int LorY)
    {
        int I = threadIdx.x + blockIdx.x*NCRIT;
        int CJ_start = offMlt[ptr_off+blockIdx.x];
        int Nmlt     = offMlt[ptr_off+blockIdx.x+1] - CJ_start;


        REAL xi, yi, zi,
             dx, dy, dz; 
        REAL a[Nm];

        __shared__ REAL xc_sh[BSZ],
                        yc_sh[BSZ],
                        zc_sh[BSZ];
        __shared__ int Index_sh[(P+1)*(P+1)*(P+1)];

        for (int ind=0; ind<((P+1)*(P+1)*(P+1)-1)/BSZ; ind++)
        {
            Index_sh[ind*BSZ + threadIdx.x] = Index[ind*BSZ + threadIdx.x];    
        }

        int ind = ((P+1)*(P+1)*(P+1)-1)/BSZ;
        if (threadIdx.x<(P+1)*(P+1)*(P+1)-BSZ*ind)
        {
            Index_sh[ind*BSZ + threadIdx.x] = Index[ind*BSZ + threadIdx.x];
        }
        int i;


        for (int iblock=0; iblock<BpT; iblock++)
        {
            i  = I + iblock*BSZ;
            xi = xt[i];
            yi = yt[i];
            zi = zt[i];
            
            REAL K = 0., V = 0.;

            for(int jblock=0; jblock<(Nmlt-1)/BSZ; jblock++)
            {
                __syncthreads();
                xc_sh[threadIdx.x] = xc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
                yc_sh[threadIdx.x] = yc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
                zc_sh[threadIdx.x] = zc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int j=0; j<BSZ; j++)
                    {
                        dx = xi - xc_sh[j];
                        dy = yi - yc_sh[j];
                        dz = zi - zc_sh[j];
                        getCoeff(a, dx, dy, dz, 
                                kappa, Index_sh, LorY);
                        multipole(K, V, M, Md, a,
                                CJ_start, jblock, j);
                    }
                }
            } 

            __syncthreads();
            int jblock = (Nmlt-1)/BSZ;
            xc_sh[threadIdx.x] = xc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
            yc_sh[threadIdx.x] = yc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
            zc_sh[threadIdx.x] = zc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
            __syncthreads();
            
            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                for (int j=0; j<Nmlt-(jblock*BSZ); j++)
                {
                    dx = xi - xc_sh[j];
                    dy = yi - yc_sh[j];
                    dz = zi - zc_sh[j];
                    getCoeff(a, dx, dy, dz, 
                            kappa, Index_sh, LorY);
                    multipole(K, V, M, Md, a,
                            CJ_start, jblock, j);
                }
            }

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                K_gpu[i] += K;
                V_gpu[i] += V; 
            }
        }
        
    }
    
    __global__ void M2PKt(REAL *Ktx_gpu, REAL *Kty_gpu, REAL *Ktz_gpu, 
                        int *offMlt, int *sizeTar, REAL *xc, REAL *yc, REAL *zc, 
                        REAL *M, REAL *xt, REAL *yt, REAL *zt,
                        int *Index, int ptr_off, int ptr_lst, REAL kappa, int BpT, int NCRIT, int LorY)
    {
        int I = threadIdx.x + blockIdx.x*NCRIT;
        int CJ_start = offMlt[ptr_off+blockIdx.x];
        int Nmlt     = offMlt[ptr_off+blockIdx.x+1] - CJ_start;


        REAL xi, yi, zi,
             dx, dy, dz; 
        REAL ax[Nm], ay[Nm], az[Nm];

        __shared__ REAL xc_sh[BSZ],
                        yc_sh[BSZ],
                        zc_sh[BSZ];
        __shared__ int Index_sh[(P+1)*(P+1)*(P+1)];

        for (int ind=0; ind<((P+1)*(P+1)*(P+1)-1)/BSZ; ind++)
        {
            Index_sh[ind*BSZ + threadIdx.x] = Index[ind*BSZ + threadIdx.x];    
        }

        int ind = ((P+1)*(P+1)*(P+1)-1)/BSZ;
        if (threadIdx.x<(P+1)*(P+1)*(P+1)-BSZ*ind)
        {
            Index_sh[ind*BSZ + threadIdx.x] = Index[ind*BSZ + threadIdx.x];
        }
        int i;


        for (int iblock=0; iblock<BpT; iblock++)
        {
            i  = I + iblock*BSZ;
            xi = xt[i];
            yi = yt[i];
            zi = zt[i];
            
            REAL Ktx = 0., Kty = 0., Ktz = 0.;

            for(int jblock=0; jblock<(Nmlt-1)/BSZ; jblock++)
            {
                __syncthreads();
                xc_sh[threadIdx.x] = xc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
                yc_sh[threadIdx.x] = yc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
                zc_sh[threadIdx.x] = zc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int j=0; j<BSZ; j++)
                    {
                        for (int ii=0; ii<Nm; ii++)
                        {
                            ax[ii] = 0.;
                            ay[ii] = 0.;
                            az[ii] = 0.;
                        }

                        dx = xi - xc_sh[j];
                        dy = yi - yc_sh[j];
                        dz = zi - zc_sh[j];
                        getCoeff_shift(ax, ay, az, dx, dy, dz, 
                                kappa, Index_sh, LorY);
                        multipoleKt(Ktx, Kty, Ktz, M, ax, ay, az,
                                CJ_start, jblock, j);
                    }
                }
            } 

            __syncthreads();
            int jblock = (Nmlt-1)/BSZ;
            xc_sh[threadIdx.x] = xc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
            yc_sh[threadIdx.x] = yc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
            zc_sh[threadIdx.x] = zc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
            __syncthreads();
            
            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                for (int j=0; j<Nmlt-(jblock*BSZ); j++)
                {
                    for (int ii=0; ii<Nm; ii++)
                    {
                        ax[ii] = 0.;
                        ay[ii] = 0.;
                        az[ii] = 0.;
                    }

                    dx = xi - xc_sh[j];
                    dy = yi - yc_sh[j];
                    dz = zi - zc_sh[j];
                    getCoeff_shift(ax, ay, az, dx, dy, dz, 
                            kappa, Index_sh, LorY);
                    multipoleKt(Ktx, Kty, Ktz, M, ax, ay, az,
                            CJ_start, jblock, j);
                }
            }

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                Ktx_gpu[i] += Ktx;
                Kty_gpu[i] += Kty;
                Ktz_gpu[i] += Ktz;
            }
        }
        
    }
    

    __global__ void P2P(REAL *K_gpu, REAL *V_gpu, int *offSrc, int *offTwg, int *P2P_list, int *sizeTar, int *k, 
                        REAL *xj, REAL *yj, REAL *zj, REAL *m, REAL *mx, REAL *my, REAL *mz, REAL *mKc, REAL *mVc, 
                        REAL *xt, REAL *yt, REAL *zt, REAL *Area, REAL *sglInt, REAL *vertex, 
                        int ptr_off, int ptr_lst, int LorY, REAL kappa, REAL threshold, 
                        int BpT, int NCRIT, REAL K_diag, int *AI_int_gpu, REAL *Xsk, REAL *Wsk)
    {
        int I = threadIdx.x + blockIdx.x*NCRIT;
        int list_start = offTwg[ptr_off+blockIdx.x];
        int list_end   = offTwg[ptr_off+blockIdx.x+1];
        
        REAL xi, yi, zi, dx, dy, dz, r, auxK, auxV;

        __shared__ REAL ver_sh[9*BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ], sglInt_sh[BSZ],
                        m_sh[BSZ], mx_sh[BSZ], my_sh[BSZ], mz_sh[BSZ], mKc_sh[BSZ], 
                        mVc_sh[BSZ], Xsk_sh[K_fine*3], Wsk_sh[K_fine];


        if (threadIdx.x<K_fine*3)
        {
            Xsk_sh[threadIdx.x] = Xsk[threadIdx.x];
            if (threadIdx.x<K_fine)
                Wsk_sh[threadIdx.x] = Wsk[threadIdx.x];
        }
        __syncthreads();

        int i, same, near, CJ_start, Nsrc, CJ;

        for (int iblock=0; iblock<BpT; iblock++)
        {
            REAL sum_K = 0., sum_V = 0.;
            i  = I + iblock*BSZ;
            xi = xt[i];
            yi = yt[i];
            zi = zt[i];
            int an_counter = 0;

            for (int lst=list_start; lst<list_end; lst++)
            {
                CJ = P2P_list[ptr_lst+lst];
                CJ_start = offSrc[CJ];
                Nsrc = offSrc[CJ+1] - CJ_start;

                for(int jblock=0; jblock<(Nsrc-1)/BSZ; jblock++)
                {
                    __syncthreads();
                    xj_sh[threadIdx.x] = xj[CJ_start + jblock*BSZ + threadIdx.x];
                    yj_sh[threadIdx.x] = yj[CJ_start + jblock*BSZ + threadIdx.x];
                    zj_sh[threadIdx.x] = zj[CJ_start + jblock*BSZ + threadIdx.x];
                    m_sh[threadIdx.x]  = m[CJ_start + jblock*BSZ + threadIdx.x];
                    mx_sh[threadIdx.x] = mx[CJ_start + jblock*BSZ + threadIdx.x];
                    my_sh[threadIdx.x] = my[CJ_start + jblock*BSZ + threadIdx.x];
                    mz_sh[threadIdx.x] = mz[CJ_start + jblock*BSZ + threadIdx.x];
                    mKc_sh[threadIdx.x] = mKc[CJ_start + jblock*BSZ + threadIdx.x];
                    mVc_sh[threadIdx.x] = mVc[CJ_start + jblock*BSZ + threadIdx.x];
                    A_sh[threadIdx.x]  = Area[CJ_start + jblock*BSZ + threadIdx.x];
                    sglInt_sh[threadIdx.x]  = sglInt[CJ_start + jblock*BSZ + threadIdx.x];
                    k_sh[threadIdx.x]  = k[CJ_start + jblock*BSZ + threadIdx.x];

                    for (int vert=0; vert<9; vert++)
                    {
                        ver_sh[9*threadIdx.x+vert] = vertex[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
                    }
                    __syncthreads();

                    if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                    {
                        for (int j=0; j<BSZ; j++)
                        {
                            dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])  *0.333333333333333333;
                            dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])*0.333333333333333333;
                            dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])*0.333333333333333333;
                            r  = 1/(dx*dx + dy*dy + dz*dz); // r is 1/r!!!
                            same = (r>1e12);
                            near = ((2*A_sh[j]*r) > threshold*threshold);
                            auxV = 0.;
                            auxK = 0.;
                           
                            if (near==0)
                            {
                                dx = xi - xj_sh[j];
                                dy = yi - yj_sh[j];
                                dz = zi - zj_sh[j];
                                r = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!!
                                if (LorY==2)
                                {
                                    auxV = exp(-kappa*1/r)*r;
                                    auxK = (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*auxV*(r)*(kappa+r);
                                    auxV *= m_sh[j];

                                }
                                if (LorY==1)
                                {
                                    auxV =  m_sh[j]*r;
                                    auxK = (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*(r*r*r);
                                }
                            }
                            
                            if ( (near==1) && (k_sh[j]==0))
                            {
                                if (same==1)
                                {
                                    auxK = K_diag;
                                    auxV = sglInt_sh[j];
                                }
                                else
                                {
                                    GQ_fine(auxK, auxV, ver_sh, 9*j, xi, yi, zi, kappa, Xsk_sh, Wsk_sh, A_sh, LorY);
                                }

                                auxV *= mVc_sh[j];
                                auxK *= mKc_sh[j]; 
                                an_counter += 1;
                            }
                            
                            sum_V += auxV;
                            sum_K += auxK;
                        }
                    }
                }
                __syncthreads();
                int jblock = (Nsrc-1)/BSZ;
                if (jblock*BSZ + threadIdx.x < Nsrc)
                {
                    xj_sh[threadIdx.x] = xj[CJ_start + jblock*BSZ + threadIdx.x];
                    yj_sh[threadIdx.x] = yj[CJ_start + jblock*BSZ + threadIdx.x];
                    zj_sh[threadIdx.x] = zj[CJ_start + jblock*BSZ + threadIdx.x];
                    m_sh[threadIdx.x] = m[CJ_start + jblock*BSZ + threadIdx.x];
                    mx_sh[threadIdx.x] = mx[CJ_start + jblock*BSZ + threadIdx.x];
                    my_sh[threadIdx.x] = my[CJ_start + jblock*BSZ + threadIdx.x];
                    mz_sh[threadIdx.x] = mz[CJ_start + jblock*BSZ + threadIdx.x];
                    mKc_sh[threadIdx.x] = mKc[CJ_start + jblock*BSZ + threadIdx.x];
                    mVc_sh[threadIdx.x] = mVc[CJ_start + jblock*BSZ + threadIdx.x];
                    A_sh[threadIdx.x] = Area[CJ_start + jblock*BSZ + threadIdx.x];
                    sglInt_sh[threadIdx.x] = sglInt[CJ_start + jblock*BSZ + threadIdx.x];
                    k_sh[threadIdx.x] = k[CJ_start + jblock*BSZ + threadIdx.x];

                    for (int vert=0; vert<9; vert++)
                    {
                        ver_sh[9*threadIdx.x+vert] = vertex[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
                    }
                }
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int j=0; j<Nsrc-(jblock*BSZ); j++)
                    {
                        dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])  *0.3333333333333333333;
                        dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])*0.3333333333333333333;
                        dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])*0.3333333333333333333;
                        r  = 1/(dx*dx + dy*dy + dz*dz); // r is 1/r!!!
                        same = (r>1e12);
                        near = ((2*A_sh[j]*r) > threshold*threshold);
                        auxV = 0.;
                        auxK = 0.;

                        if (near==0)
                        {
                            dx = xi - xj_sh[j];
                            dy = yi - yj_sh[j];
                            dz = zi - zj_sh[j];
                            r = rsqrt(dx*dx + dy*dy + dz*dz);  // r is 1/r!!!

                            if (LorY==2)
                            {
                                auxV = exp(-kappa*1/r)*r;
                                auxK = (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*auxV*(r)*(kappa+r);
                                auxV *= m_sh[j];
                            }
                            if (LorY==1)
                            {
                                auxV = m_sh[j]*r;
                                auxK = (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*(r*r*r);
                            }
                        }
                        
                        if ( (near==1) && (k_sh[j]==0))
                        {
                            if (same==1)
                            {
                                auxK = K_diag;
                                auxV = sglInt_sh[j];
                            }
                            else
                            {
                                GQ_fine(auxK, auxV, ver_sh, 9*j, xi, yi, zi, kappa, Xsk_sh, Wsk_sh, A_sh, LorY);
                            }

                            auxV *= mVc_sh[j];
                            auxK *= mKc_sh[j];
                            an_counter += 1;
                        }
                       
                        sum_V += auxV;
                        sum_K += auxK;
                    }
                }
            }
        
            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                K_gpu[i] += sum_K;
                V_gpu[i] += sum_V; 

                AI_int_gpu[i] = an_counter;
            }
        }
    }
   
    __global__ void P2PKt(REAL *Ktx_gpu, REAL *Kty_gpu, REAL *Ktz_gpu, int *offSrc, int *offTwg, int *P2P_list, int *sizeTar, int *k, 
                        REAL *xj, REAL *yj, REAL *zj, REAL *m, REAL *mKtc,
                        REAL *xt, REAL *yt, REAL *zt, REAL *Area, REAL *vertex, 
                        int ptr_off, int ptr_lst, int LorY, REAL kappa, REAL threshold, 
                        int BpT, int NCRIT, int *AI_int_gpu, REAL *Xsk, REAL *Wsk)
    {
        int I = threadIdx.x + blockIdx.x*NCRIT;
        int list_start = offTwg[ptr_off+blockIdx.x];
        int list_end   = offTwg[ptr_off+blockIdx.x+1];
        
        REAL xi, yi, zi, dx, dy, dz, r, auxKtx, auxKty, auxKtz;

        __shared__ REAL ver_sh[9*BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ],
                        m_sh[BSZ], mKtc_sh[BSZ], 
                        Xsk_sh[K_fine*3], Wsk_sh[K_fine];


        if (threadIdx.x<K_fine*3)
        {
            Xsk_sh[threadIdx.x] = Xsk[threadIdx.x];
            if (threadIdx.x<K_fine)
                Wsk_sh[threadIdx.x] = Wsk[threadIdx.x];
        }
        __syncthreads();

        int i, same, near, CJ_start, Nsrc, CJ;

        for (int iblock=0; iblock<BpT; iblock++)
        {
            REAL sum_Ktx = 0., sum_Kty = 0., sum_Ktz = 0.;
            i  = I + iblock*BSZ;
            xi = xt[i];
            yi = yt[i];
            zi = zt[i];
            int an_counter = 0;

            for (int lst=list_start; lst<list_end; lst++)
            {
                CJ = P2P_list[ptr_lst+lst];
                CJ_start = offSrc[CJ];
                Nsrc = offSrc[CJ+1] - CJ_start;

                for(int jblock=0; jblock<(Nsrc-1)/BSZ; jblock++)
                {
                    __syncthreads();
                    xj_sh[threadIdx.x]   = xj[CJ_start + jblock*BSZ + threadIdx.x];
                    yj_sh[threadIdx.x]   = yj[CJ_start + jblock*BSZ + threadIdx.x];
                    zj_sh[threadIdx.x]   = zj[CJ_start + jblock*BSZ + threadIdx.x];
                    m_sh[threadIdx.x]    = m[CJ_start + jblock*BSZ + threadIdx.x];
                    mKtc_sh[threadIdx.x] = mKtc[CJ_start + jblock*BSZ + threadIdx.x];
                    A_sh[threadIdx.x]    = Area[CJ_start + jblock*BSZ + threadIdx.x];
                    k_sh[threadIdx.x]    = k[CJ_start + jblock*BSZ + threadIdx.x];

                    for (int vert=0; vert<9; vert++)
                    {
                        ver_sh[9*threadIdx.x+vert] = vertex[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
                    }
                    __syncthreads();

                    if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                    {
                        for (int j=0; j<BSZ; j++)
                        {
                            dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])  *0.333333333333333333;
                            dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])*0.333333333333333333;
                            dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])*0.333333333333333333;
                            r  = 1/(dx*dx + dy*dy + dz*dz); // r is 1/r!!!
                            same = (r>1e12);
                            near = ((2*A_sh[j]*r) > threshold*threshold);
                            auxKtx = 0.;
                            auxKty = 0.;
                            auxKtz = 0.;
                           
                            if (near==0)
                            {
                                dx = xi - xj_sh[j];
                                dy = yi - yj_sh[j];
                                dz = zi - zj_sh[j];
                                r = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!!
                                if (LorY==2)
                                {
                                    auxKtx  = -m_sh[j]*exp(-kappa*1/r)*r*r*(kappa+r);
                                    auxKty  = auxKtx*dy;
                                    auxKtz  = auxKtx*dz;
                                    auxKtx *= dx;
                                }
                                if (LorY==1)
                                {
                                    auxKtx  = -m_sh[j]*r*r*r;
                                    auxKty  = auxKtx*dy;
                                    auxKtz  = auxKtx*dz;
                                    auxKtx *= dx;
                                }
                            }
                            
                            if ( (near==1) && (k_sh[j]==0))
                            {
                                if (same==1)
                                {
                                    auxKtx = 0.0;
                                    auxKty = 0.0;
                                    auxKtz = 0.0;
                                }
                                else
                                {
                                    GQ_fineKt(auxKtx, auxKty, auxKtz, ver_sh, 9*j, xi, yi, zi, kappa, Xsk_sh, Wsk_sh, A_sh, LorY);
                                }

                                auxKtx *= mKtc_sh[j];
                                auxKty *= mKtc_sh[j];
                                auxKtz *= mKtc_sh[j];
                                an_counter += 1;
                            }
                            
                            sum_Ktx += auxKtx;
                            sum_Kty += auxKty;
                            sum_Ktz += auxKtz;
                        }
                    }
                }
                __syncthreads();
                int jblock = (Nsrc-1)/BSZ;
                if (jblock*BSZ + threadIdx.x < Nsrc)
                {
                    xj_sh[threadIdx.x] = xj[CJ_start + jblock*BSZ + threadIdx.x];
                    yj_sh[threadIdx.x] = yj[CJ_start + jblock*BSZ + threadIdx.x];
                    zj_sh[threadIdx.x] = zj[CJ_start + jblock*BSZ + threadIdx.x];
                    m_sh[threadIdx.x] = m[CJ_start + jblock*BSZ + threadIdx.x];
                    mKtc_sh[threadIdx.x] = mKtc[CJ_start + jblock*BSZ + threadIdx.x];
                    A_sh[threadIdx.x] = Area[CJ_start + jblock*BSZ + threadIdx.x];
                    k_sh[threadIdx.x] = k[CJ_start + jblock*BSZ + threadIdx.x];

                    for (int vert=0; vert<9; vert++)
                    {
                        ver_sh[9*threadIdx.x+vert] = vertex[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
                    }
                }
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int j=0; j<Nsrc-(jblock*BSZ); j++)
                    {
                        dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])  *0.3333333333333333333;
                        dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])*0.3333333333333333333;
                        dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])*0.3333333333333333333;
                        r  = 1/(dx*dx + dy*dy + dz*dz); // r is 1/r!!!
                        same = (r>1e12);
                        near = ((2*A_sh[j]*r) > threshold*threshold);
                        auxKtx = 0.;
                        auxKty = 0.;
                        auxKtz = 0.;

                        if (near==0)
                        {
                            dx = xi - xj_sh[j];
                            dy = yi - yj_sh[j];
                            dz = zi - zj_sh[j];
                            r = rsqrt(dx*dx + dy*dy + dz*dz);  // r is 1/r!!!

                            if (LorY==2)
                            {
                                auxKtx  = -m_sh[j]*exp(-kappa*1/r)*r*r*(kappa+r);
                                auxKty  = auxKtx*dy;
                                auxKtz  = auxKtx*dz;
                                auxKtx *= dx;
                            }
                            if (LorY==1)
                            {
                                auxKtx  = -m_sh[j]*r*r*r;
                                auxKty  = auxKtx*dy;
                                auxKtz  = auxKtx*dz;
                                auxKtx *= dx;
                            }
                        }
                        
                        if ( (near==1) && (k_sh[j]==0))
                        {
                            if (same==1)
                            {
                                auxKtx = 0.0;
                                auxKty = 0.0;
                                auxKtz = 0.0;
                            }
                            else
                            {
                                GQ_fineKt(auxKtx, auxKty, auxKtz, ver_sh, 9*j, xi, yi, zi, kappa, Xsk_sh, Wsk_sh, A_sh, LorY);
                            }

                            auxKtx *= mKtc_sh[j];
                            auxKty *= mKtc_sh[j];
                            auxKtz *= mKtc_sh[j];
                            an_counter += 1;
                        }
                       
                        sum_Ktx += auxKtx;
                        sum_Kty += auxKty;
                        sum_Ktz += auxKtz;
                    }
                }
            }
        
            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                Ktx_gpu[i] += sum_Ktx;
                Kty_gpu[i] += sum_Kty;
                Ktz_gpu[i] += sum_Ktz;

                AI_int_gpu[i] = an_counter;
            }
        }
    }

    __global__ void get_phir(REAL *phir, REAL *xq, REAL *yq, REAL *zq,
                            REAL *m, REAL *mx, REAL *my, REAL *mz, REAL *mKc, REAL *mVc, 
                            REAL *xj, REAL *yj, REAL *zj, REAL *Area, int *k, REAL *vertex, 
                            int Nj, int Nq, int K, REAL *xk, REAL *wk, 
                            REAL threshold, int *AI_int_gpu, int Nk, REAL *Xsk, REAL *Wsk)
    {
        int i = threadIdx.x + blockIdx.x*BSZ;
        REAL xi, yi, zi, dx, dy, dz, r;
        int jblock, triangle;

        __shared__ REAL ver_sh[9*BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ],
                        m_sh[BSZ], mx_sh[BSZ], my_sh[BSZ], mz_sh[BSZ], mKc_sh[BSZ],
                        mVc_sh[BSZ];



        REAL sum_V = 0., sum_K = 0.;
        xi = xq[i];
        yi = yq[i];
        zi = zq[i];
        int an_counter = 0;

        for(jblock=0; jblock<(Nj-1)/BSZ; jblock++)
        {   
            __syncthreads();
            xj_sh[threadIdx.x] = xj[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = yj[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zj[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mKc_sh[threadIdx.x] = mKc[jblock*BSZ + threadIdx.x];
            mVc_sh[threadIdx.x] = mVc[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[(jblock*BSZ + threadIdx.x)];
            for (int vert=0; vert<9; vert++)
            {
                triangle = jblock*BSZ+threadIdx.x;
                ver_sh[9*threadIdx.x+vert] = vertex[9*triangle+vert];
            }
            __syncthreads();
            
            for (int j=0; j<BSZ; j++)
            {
                dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])/3;
                dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])/3;
                dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])/3;
                r  = sqrt(dx*dx + dy*dy + dz*dz);

                if((sqrt(2*A_sh[j])/r) < threshold)
                {
                    dx = xi - xj_sh[j];
                    dy = yi - yj_sh[j];
                    dz = zi - zj_sh[j];
                    r = sqrt(dx*dx + dy*dy + dz*dz);
                    sum_V  += m_sh[j]/r; 
                    sum_K += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                }
                else if(k_sh[j]==0)
                {
                    REAL PHI_K = 0.;
                    REAL PHI_V = 0.;

                    GQ_fine(PHI_K, PHI_V, ver_sh, 9*j, xi, yi, zi, 1e-15, Xsk, Wsk, A_sh, 1);
                    //REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                    //                 ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                    //                 ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};
                    //SA(PHI_K, PHI_V, panel, xi, yi, zi, 
                    //   1., 1., 1e-15, 0, xk, wk, 9, 1);
        
                    sum_V += PHI_V * mVc_sh[j];
                    sum_K += PHI_K * mKc_sh[j];
                    an_counter += 1;
                }
            }
        }
    
        __syncthreads();
        jblock = (Nj-1)/BSZ;
        if (threadIdx.x<Nj-jblock*BSZ)
        {
            xj_sh[threadIdx.x] = xj[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = yj[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zj[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mKc_sh[threadIdx.x] = mKc[jblock*BSZ + threadIdx.x];
            mVc_sh[threadIdx.x] = mVc[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[jblock*BSZ + threadIdx.x];

            for (int vert=0; vert<9; vert++)
            {
                triangle = jblock*BSZ+threadIdx.x;
                ver_sh[9*threadIdx.x+vert] = vertex[9*triangle+vert];
            }
        }
        __syncthreads();


        for (int j=0; j<Nj-(jblock*BSZ); j++)
        {
            dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])/3;
            dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])/3;
            dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])/3;
            r  = sqrt(dx*dx + dy*dy + dz*dz);

            if (i<Nq)
            {
                if ((sqrt(2*A_sh[j])/r) < threshold)
                {
                    dx = xi - xj_sh[j];
                    dy = yi - yj_sh[j];
                    dz = zi - zj_sh[j];
                    r = sqrt(dx*dx + dy*dy + dz*dz);
                    sum_V  += m_sh[j]/r; 
                    sum_K += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                }

                else if(k_sh[j]==0)
                {
                    REAL PHI_K = 0.;
                    REAL PHI_V = 0.;

                    GQ_fine(PHI_K, PHI_V, ver_sh, 9*j, xi, yi, zi, 1e-15, Xsk, Wsk, A_sh, 1);
        
                    sum_V += PHI_V * mVc_sh[j];
                    sum_K += PHI_K * mKc_sh[j];
                    
                    an_counter += 1;
                }
            }
        }
       
        if (i<Nq)
        {
            phir[i] = (-sum_K + sum_V)/(4*M_PI);
            AI_int_gpu[i] = an_counter;
        }
    }    

__global__ void get_dphirdr(REAL *dphir_x, REAL *dphir_y, REAL *dphir_z, REAL *xq, REAL *yq, REAL *zq,
                            REAL *m, REAL *mx, REAL *my, REAL *mz, REAL *mKc, REAL *mVc, 
                            REAL *xj, REAL *yj, REAL *zj, REAL *Area, int *k, REAL *vertex, 
                            int Nj, int Nq, int K, REAL *xk, REAL *wk, 
                            REAL threshold, int *AI_int_gpu, int Nk, REAL *Xsk, REAL *Wsk)
    {
        int i = threadIdx.x + blockIdx.x*BSZ;
        REAL xi, yi, zi, dx, dy, dz, r, r3;
        int jblock, triangle;

        __shared__ REAL ver_sh[9*BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ],
                        m_sh[BSZ], mx_sh[BSZ], my_sh[BSZ], mz_sh[BSZ], mKc_sh[BSZ],
                        mVc_sh[BSZ];



        REAL sum_Vx = 0., sum_Kx = 0.;
        REAL sum_Vy = 0., sum_Ky = 0.;
        REAL sum_Vz = 0., sum_Kz = 0.;
        xi = xq[i];
        yi = yq[i];
        zi = zq[i];
        int an_counter = 0;

        for(jblock=0; jblock<(Nj-1)/BSZ; jblock++)
        {   
            __syncthreads();
            xj_sh[threadIdx.x] = xj[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = yj[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zj[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mKc_sh[threadIdx.x] = mKc[jblock*BSZ + threadIdx.x];
            mVc_sh[threadIdx.x] = mVc[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[(jblock*BSZ + threadIdx.x)];
            for (int vert=0; vert<9; vert++)
            {
                triangle = jblock*BSZ+threadIdx.x;
                ver_sh[9*threadIdx.x+vert] = vertex[9*triangle+vert];
            }
            __syncthreads();
            
            for (int j=0; j<BSZ; j++)
            {
                dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])/3;
                dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])/3;
                dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])/3;
                r  = sqrt(dx*dx + dy*dy + dz*dz);

                if((sqrt(2*A_sh[j])/r) < threshold)
                {
                    dx = xi - xj_sh[j];
                    dy = yi - yj_sh[j];
                    dz = zi - zj_sh[j];
                    r  = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r
                    r3 = r*r*r;

                    sum_Vx -= m_sh[j]*dx*r3; 
                    sum_Vy -= m_sh[j]*dy*r3; 
                    sum_Vz -= m_sh[j]*dz*r3; 
                    sum_Kx += mx_sh[j]*r3 - 3*dx*(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*r3*r*r;
                    sum_Ky += my_sh[j]*r3 - 3*dy*(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*r3*r*r;
                    sum_Kz += mz_sh[j]*r3 - 3*dz*(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*r3*r*r;
                }

                else if(k_sh[j]==0)
                {
                    REAL dPHI_Kx = 0.;
                    REAL dPHI_Vx = 0.;
                    REAL dPHI_Ky = 0.;
                    REAL dPHI_Vy = 0.;
                    REAL dPHI_Kz = 0.;
                    REAL dPHI_Vz = 0.;

                    GQ_fine_derivative(dPHI_Kx, dPHI_Vx, dPHI_Ky, dPHI_Vy, dPHI_Kz, dPHI_Vz, 
                                        ver_sh, 9*j, xi, yi, zi, 1e-15, Xsk, Wsk, A_sh, 1);
                    //REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                    //                 ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                    //                 ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};
                    //SA(PHI_K, PHI_V, panel, xi, yi, zi, 
                    //   1., 1., 1e-15, 0, xk, wk, 9, 1);
        
                    sum_Vx += dPHI_Vx * mVc_sh[j];
                    sum_Vy += dPHI_Vy * mVc_sh[j];
                    sum_Vz += dPHI_Vz * mVc_sh[j];
                    sum_Kx += dPHI_Kx * mKc_sh[j];
                    sum_Ky += dPHI_Ky * mKc_sh[j];
                    sum_Kz += dPHI_Kz * mKc_sh[j];
                    an_counter += 1;
                }
            }
        }
    
        __syncthreads();
        jblock = (Nj-1)/BSZ;
        if (threadIdx.x<Nj-jblock*BSZ)
        {
            xj_sh[threadIdx.x] = xj[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = yj[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zj[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mKc_sh[threadIdx.x] = mKc[jblock*BSZ + threadIdx.x];
            mVc_sh[threadIdx.x] = mVc[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[jblock*BSZ + threadIdx.x];

            for (int vert=0; vert<9; vert++)
            {
                triangle = jblock*BSZ+threadIdx.x;
                ver_sh[9*threadIdx.x+vert] = vertex[9*triangle+vert];
            }
        }
        __syncthreads();


        for (int j=0; j<Nj-(jblock*BSZ); j++)
        {
            dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])/3;
            dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])/3;
            dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])/3;
            r  = sqrt(dx*dx + dy*dy + dz*dz);

            if (i<Nq)
            {
                if ((sqrt(2*A_sh[j])/r) < threshold)
                {
                    dx = xi - xj_sh[j];
                    dy = yi - yj_sh[j];
                    dz = zi - zj_sh[j];
                    r  = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r
                    r3 = r*r*r;

                    sum_Vx -= m_sh[j]*dx*r3; 
                    sum_Vy -= m_sh[j]*dy*r3; 
                    sum_Vz -= m_sh[j]*dz*r3; 
                    sum_Kx += mx_sh[j]*r3 - 3*dx*(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*r3*r*r;
                    sum_Ky += my_sh[j]*r3 - 3*dy*(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*r3*r*r;
                    sum_Kz += mz_sh[j]*r3 - 3*dz*(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*r3*r*r;

                }

                else if(k_sh[j]==0)
                {
                    REAL dPHI_Kx = 0.;
                    REAL dPHI_Vx = 0.;
                    REAL dPHI_Ky = 0.;
                    REAL dPHI_Vy = 0.;
                    REAL dPHI_Kz = 0.;
                    REAL dPHI_Vz = 0.;

                    GQ_fine_derivative(dPHI_Kx, dPHI_Vx, dPHI_Ky, dPHI_Vy, dPHI_Kz, dPHI_Vz, 
                                        ver_sh, 9*j, xi, yi, zi, 1e-15, Xsk, Wsk, A_sh, 1);
        
                    sum_Vx += dPHI_Vx * mVc_sh[j];
                    sum_Vy += dPHI_Vy * mVc_sh[j];
                    sum_Vz += dPHI_Vz * mVc_sh[j];
                    sum_Kx += dPHI_Kx * mKc_sh[j];
                    sum_Ky += dPHI_Ky * mKc_sh[j];
                    sum_Kz += dPHI_Kz * mKc_sh[j];
                    an_counter += 1;
                }
            }
        }
       
        if (i<Nq)
        {
            dphir_x[i] = (-sum_Kx + sum_Vx)/(4*M_PI);
            dphir_y[i] = (-sum_Ky + sum_Vy)/(4*M_PI);
            dphir_z[i] = (-sum_Kz + sum_Vz)/(4*M_PI);
            AI_int_gpu[i] = an_counter;
        }
    }

__global__ void get_d2phirdr2(REAL *ddphir_xx, REAL *ddphir_xy, REAL *ddphir_xz, 
                            REAL *ddphir_yx, REAL *ddphir_yy, REAL *ddphir_yz,
                            REAL *ddphir_zx, REAL *ddphir_zy, REAL *ddphir_zz,
                            REAL *xq, REAL *yq, REAL *zq,
                            REAL *m, REAL *mx, REAL *my, REAL *mz, REAL *mKc, REAL *mVc, 
                            REAL *xj, REAL *yj, REAL *zj, REAL *Area, int *k, REAL *vertex, 
                            int Nj, int Nq, int K, REAL *xk, REAL *wk, 
                            REAL threshold, int *AI_int_gpu, int Nk, REAL *Xsk, REAL *Wsk)
    {
        int i = threadIdx.x + blockIdx.x*BSZ;
        REAL xi, yi, zi, dx, dy, dz, R, R2, R3;
        int jblock, triangle;

        __shared__ REAL ver_sh[9*BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ],
                        m_sh[BSZ], mx_sh[BSZ], my_sh[BSZ], mz_sh[BSZ], mKc_sh[BSZ],
                        mVc_sh[BSZ];



        REAL sum_Vxx = 0., sum_Kxx = 0.;
        REAL sum_Vxy = 0., sum_Kxy = 0.;
        REAL sum_Vxz = 0., sum_Kxz = 0.;
        REAL sum_Vyx = 0., sum_Kyx = 0.;
        REAL sum_Vyy = 0., sum_Kyy = 0.;
        REAL sum_Vyz = 0., sum_Kyz = 0.;
        REAL sum_Vzx = 0., sum_Kzx = 0.;
        REAL sum_Vzy = 0., sum_Kzy = 0.;
        REAL sum_Vzz = 0., sum_Kzz = 0.;

        xi = xq[i];
        yi = yq[i];
        zi = zq[i];
        int an_counter = 0;

        for(jblock=0; jblock<(Nj-1)/BSZ; jblock++)
        {   
            __syncthreads();
            xj_sh[threadIdx.x] = xj[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = yj[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zj[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mKc_sh[threadIdx.x] = mKc[jblock*BSZ + threadIdx.x];
            mVc_sh[threadIdx.x] = mVc[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[(jblock*BSZ + threadIdx.x)];
            for (int vert=0; vert<9; vert++)
            {
                triangle = jblock*BSZ+threadIdx.x;
                ver_sh[9*threadIdx.x+vert] = vertex[9*triangle+vert];
            }
            __syncthreads();
            
            for (int j=0; j<BSZ; j++)
            {
                dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])/3;
                dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])/3;
                dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])/3;
                R  = sqrt(dx*dx + dy*dy + dz*dz);

                if((sqrt(2*A_sh[j])/R) < threshold)
                {
                    dx = xi - xj_sh[j];
                    dy = yi - yj_sh[j];
                    dz = zi - zj_sh[j];
                    R  = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r
                    R2 = R*R;
                    R3 = R2*R;

                    sum_Vxx += m_sh[j]*R3*(-1 + 3*dx*dx*R2);
                    sum_Vxy += m_sh[j]*R3*3*dx*dy*R2;
                    sum_Vxz += m_sh[j]*R3*3*dx*dz*R2;
                    sum_Vyx += m_sh[j]*R3*3*dy*dx*R2;
                    sum_Vyy += m_sh[j]*R3*(-1 + 3*dy*dy*R2);
                    sum_Vyz += m_sh[j]*R3*3*dy*dz*R2;
                    sum_Vzx += m_sh[j]*R3*3*dz*dx*R2;
                    sum_Vzy += m_sh[j]*R3*3*dz*dy*R2;
                    sum_Vzz += m_sh[j]*R3*(-1 + 3*dz*dz*R2);
                    sum_Kxx -= 3*R3*R2*(2*dx*mx_sh[j]+ dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j] - 5*R2*dx*dx*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kxy -= 3*R3*R2*(dx*my_sh[j] + dy*mx_sh[j] - 5*R2*dx*dy*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kxz -= 3*R3*R2*(dx*mz_sh[j] + dz*mx_sh[j] - 5*R2*dx*dz*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kyx -= 3*R3*R2*(dy*mx_sh[j] + dx*my_sh[j] - 5*R2*dy*dx*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kyy -= 3*R3*R2*(2*dy*my_sh[j]+ dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j] - 5*R2*dy*dy*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kyz -= 3*R3*R2*(dy*mz_sh[j] + dz*my_sh[j] - 5*R2*dy*dz*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kzx -= 3*R3*R2*(dz*mx_sh[j] + dx*mz_sh[j] - 5*R2*dz*dx*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kzy -= 3*R3*R2*(dz*my_sh[j] + dy*mz_sh[j] - 5*R2*dz*dy*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kzz -= 3*R3*R2*(2*dz*mz_sh[j]+ dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j] - 5*R2*dz*dz*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));

                }

                else if(k_sh[j]==0)
                {
                    REAL ddPHI_Kxx = 0.;
                    REAL ddPHI_Vxx = 0.;
                    REAL ddPHI_Kxy = 0.;
                    REAL ddPHI_Vxy = 0.;
                    REAL ddPHI_Kxz = 0.;
                    REAL ddPHI_Vxz = 0.;
                    REAL ddPHI_Kyx = 0.;
                    REAL ddPHI_Vyx = 0.;
                    REAL ddPHI_Kyy = 0.;
                    REAL ddPHI_Vyy = 0.;
                    REAL ddPHI_Kyz = 0.;
                    REAL ddPHI_Vyz = 0.;
                    REAL ddPHI_Kzx = 0.;
                    REAL ddPHI_Vzx = 0.;
                    REAL ddPHI_Kzy = 0.;
                    REAL ddPHI_Vzy = 0.;
                    REAL ddPHI_Kzz = 0.;
                    REAL ddPHI_Vzz = 0.;

                    GQ_fine_2derivative(ddPHI_Kxx, ddPHI_Vxx, ddPHI_Kxy, ddPHI_Vxy, ddPHI_Kxz, ddPHI_Vxz, 
                                       ddPHI_Kyx, ddPHI_Vyx, ddPHI_Kyy, ddPHI_Vyy, ddPHI_Kyz, ddPHI_Vyz,
                                       ddPHI_Kzx, ddPHI_Vzx, ddPHI_Kzy, ddPHI_Vzy, ddPHI_Kzz, ddPHI_Vzz,
                                       ver_sh, 9*j, xi, yi, zi, 1e-15, Xsk, Wsk, A_sh, 1);
                    //REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                    //                 ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                    //                 ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};
                    //SA(PHI_K, PHI_V, panel, xi, yi, zi, 
                    //   1., 1., 1e-15, 0, xk, wk, 9, 1);
        
                    sum_Vxx += ddPHI_Vxx * mVc_sh[j];
                    sum_Vxy += ddPHI_Vxy * mVc_sh[j];
                    sum_Vxz += ddPHI_Vxz * mVc_sh[j];
                    sum_Kxx += ddPHI_Kxx * mKc_sh[j];
                    sum_Kxy += ddPHI_Kxy * mKc_sh[j];
                    sum_Kxz += ddPHI_Kxz * mKc_sh[j];
                    sum_Vyx += ddPHI_Vyx * mVc_sh[j];
                    sum_Vyy += ddPHI_Vyy * mVc_sh[j];
                    sum_Vyz += ddPHI_Vyz * mVc_sh[j];
                    sum_Kyx += ddPHI_Kyx * mKc_sh[j];
                    sum_Kyy += ddPHI_Kyy * mKc_sh[j];
                    sum_Kyz += ddPHI_Kyz * mKc_sh[j];
                    sum_Vzx += ddPHI_Vzx * mVc_sh[j];
                    sum_Vzy += ddPHI_Vzy * mVc_sh[j];
                    sum_Vzz += ddPHI_Vzz * mVc_sh[j];
                    sum_Kzx += ddPHI_Kzx * mKc_sh[j];
                    sum_Kzy += ddPHI_Kzy * mKc_sh[j];
                    sum_Kzz += ddPHI_Kzz * mKc_sh[j];
                    an_counter += 1;
                }
            }
        }
    
        __syncthreads();
        jblock = (Nj-1)/BSZ;
        if (threadIdx.x<Nj-jblock*BSZ)
        {
            xj_sh[threadIdx.x] = xj[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = yj[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zj[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mKc_sh[threadIdx.x] = mKc[jblock*BSZ + threadIdx.x];
            mVc_sh[threadIdx.x] = mVc[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[jblock*BSZ + threadIdx.x];

            for (int vert=0; vert<9; vert++)
            {
                triangle = jblock*BSZ+threadIdx.x;
                ver_sh[9*threadIdx.x+vert] = vertex[9*triangle+vert];
            }
        }
        __syncthreads();


        for (int j=0; j<Nj-(jblock*BSZ); j++)
        {
            dx = xi - (ver_sh[9*j] + ver_sh[9*j+3] + ver_sh[9*j+6])/3;
            dy = yi - (ver_sh[9*j+1] + ver_sh[9*j+4] + ver_sh[9*j+7])/3;
            dz = zi - (ver_sh[9*j+2] + ver_sh[9*j+5] + ver_sh[9*j+8])/3;
            R  = sqrt(dx*dx + dy*dy + dz*dz);

            if (i<Nq)
            {
                if ((sqrt(2*A_sh[j])/R) < threshold)
                {
                    dx = xi - xj_sh[j];
                    dy = yi - yj_sh[j];
                    dz = zi - zj_sh[j];
                    R  = rsqrt(dx*dx + dy*dy + dz*dz); // r is 1/r
                    R2 = R*R;
                    R3 = R2*R;

                    sum_Vxx += m_sh[j]*R3*(-1 + 3*dx*dx*R2);
                    sum_Vxy += m_sh[j]*R3*3*dx*dy*R2;
                    sum_Vxz += m_sh[j]*R3*3*dx*dz*R2;
                    sum_Vyx += m_sh[j]*R3*3*dy*dx*R2;
                    sum_Vyy += m_sh[j]*R3*(-1 + 3*dy*dy*R2);
                    sum_Vyz += m_sh[j]*R3*3*dy*dz*R2;
                    sum_Vzx += m_sh[j]*R3*3*dz*dx*R2;
                    sum_Vzy += m_sh[j]*R3*3*dz*dy*R2;
                    sum_Vzz += m_sh[j]*R3*(-1 + 3*dz*dz*R2);
                    sum_Kxx -= 3*R3*R2*(2*dx*mx_sh[j]+ dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j] - 5*R2*dx*dx*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kxy -= 3*R3*R2*(dx*my_sh[j] + dy*mx_sh[j] - 5*R2*dx*dy*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kxz -= 3*R3*R2*(dx*mz_sh[j] + dz*mx_sh[j] - 5*R2*dx*dz*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kyx -= 3*R3*R2*(dy*mx_sh[j] + dx*my_sh[j] - 5*R2*dy*dx*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kyy -= 3*R3*R2*(2*dy*my_sh[j]+ dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j] - 5*R2*dy*dy*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kyz -= 3*R3*R2*(dy*mz_sh[j] + dz*my_sh[j] - 5*R2*dy*dz*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kzx -= 3*R3*R2*(dz*mx_sh[j] + dx*mz_sh[j] - 5*R2*dz*dx*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kzy -= 3*R3*R2*(dz*my_sh[j] + dy*mz_sh[j] - 5*R2*dz*dy*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));
                    sum_Kzz -= 3*R3*R2*(2*dz*mz_sh[j]+ dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j] - 5*R2*dz*dz*(dx*mx_sh[j]+dy*my_sh[j]+dz*mz_sh[j]));

                }

                else if(k_sh[j]==0)
                {
                    REAL ddPHI_Kxx = 0.;
                    REAL ddPHI_Vxx = 0.;
                    REAL ddPHI_Kxy = 0.;
                    REAL ddPHI_Vxy = 0.;
                    REAL ddPHI_Kxz = 0.;
                    REAL ddPHI_Vxz = 0.;
                    REAL ddPHI_Kyx = 0.;
                    REAL ddPHI_Vyx = 0.;
                    REAL ddPHI_Kyy = 0.;
                    REAL ddPHI_Vyy = 0.;
                    REAL ddPHI_Kyz = 0.;
                    REAL ddPHI_Vyz = 0.;
                    REAL ddPHI_Kzx = 0.;
                    REAL ddPHI_Vzx = 0.;
                    REAL ddPHI_Kzy = 0.;
                    REAL ddPHI_Vzy = 0.;
                    REAL ddPHI_Kzz = 0.;
                    REAL ddPHI_Vzz = 0.;

                    GQ_fine_2derivative(ddPHI_Kxx, ddPHI_Vxx, ddPHI_Kxy, ddPHI_Vxy, ddPHI_Kxz, ddPHI_Vxz, 
                                       ddPHI_Kyx, ddPHI_Vyx, ddPHI_Kyy, ddPHI_Vyy, ddPHI_Kyz, ddPHI_Vyz,
                                       ddPHI_Kzx, ddPHI_Vzx, ddPHI_Kzy, ddPHI_Vzy, ddPHI_Kzz, ddPHI_Vzz,
                                       ver_sh, 9*j, xi, yi, zi, 1e-15, Xsk, Wsk, A_sh, 1);
        
                    sum_Vxx += ddPHI_Vxx * mVc_sh[j];
                    sum_Vxy += ddPHI_Vxy * mVc_sh[j];
                    sum_Vxz += ddPHI_Vxz * mVc_sh[j];
                    sum_Kxx += ddPHI_Kxx * mKc_sh[j];
                    sum_Kxy += ddPHI_Kxy * mKc_sh[j];
                    sum_Kxz += ddPHI_Kxz * mKc_sh[j];
                    sum_Vyx += ddPHI_Vyx * mVc_sh[j];
                    sum_Vyy += ddPHI_Vyy * mVc_sh[j];
                    sum_Vyz += ddPHI_Vyz * mVc_sh[j];
                    sum_Kyx += ddPHI_Kyx * mKc_sh[j];
                    sum_Kyy += ddPHI_Kyy * mKc_sh[j];
                    sum_Kyz += ddPHI_Kyz * mKc_sh[j];
                    sum_Vzx += ddPHI_Vzx * mVc_sh[j];
                    sum_Vzy += ddPHI_Vzy * mVc_sh[j];
                    sum_Vzz += ddPHI_Vzz * mVc_sh[j];
                    sum_Kzx += ddPHI_Kzx * mKc_sh[j];
                    sum_Kzy += ddPHI_Kzy * mKc_sh[j];
                    sum_Kzz += ddPHI_Kzz * mKc_sh[j];
                    an_counter += 1;
                }
            }
        }
       
        if (i<Nq)
        {
            ddphir_xx[i] = (-sum_Kxx + sum_Vxx)/(4*M_PI);
            ddphir_xy[i] = (-sum_Kxy + sum_Vxy)/(4*M_PI);
            ddphir_xz[i] = (-sum_Kxz + sum_Vxz)/(4*M_PI);
            ddphir_yx[i] = (-sum_Kyx + sum_Vyx)/(4*M_PI);
            ddphir_yy[i] = (-sum_Kyy + sum_Vyy)/(4*M_PI);
            ddphir_yz[i] = (-sum_Kyz + sum_Vyz)/(4*M_PI);
            ddphir_zx[i] = (-sum_Kzx + sum_Vzx)/(4*M_PI);
            ddphir_zy[i] = (-sum_Kzy + sum_Vzy)/(4*M_PI);
            ddphir_zz[i] = (-sum_Kzz + sum_Vzz)/(4*M_PI);
            AI_int_gpu[i] = an_counter;
        }
    }

    __global__ void compute_RHS(REAL *F, REAL *xq, REAL *yq, REAL *zq,
                                REAL *q, REAL *xi, REAL *yi, REAL *zi,
                                int *sizeTar, int Nq, REAL E_1, 
                                int NCRIT, int BpT)
    {
        int II = threadIdx.x + blockIdx.x*NCRIT;
        int I;
        REAL x, y, z, sum;
        REAL dx, dy, dz, r;
        __shared__ REAL xq_sh[BSZ], yq_sh[BSZ], zq_sh[BSZ], q_sh[BSZ];

        for (int iblock=0; iblock<BpT; iblock++)
        {
            I = II + iblock*BSZ;
            x = xi[I];
            y = yi[I];
            z = zi[I];
            sum = 0.;

            for (int block=0; block<(Nq-1)/BSZ; block++)
            {
                __syncthreads();
                xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
                yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
                zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
                q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int i=0; i<BSZ; i++)
                    {
                        dx = xq_sh[i] - x;
                        dy = yq_sh[i] - y;
                        dz = zq_sh[i] - z;
                        r  = sqrt(dx*dx + dy*dy + dz*dz);

                        sum += q_sh[i]/(E_1*r);
                    }
                }
            }

            int block = (Nq-1)/BSZ; 
            __syncthreads();
            xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
            yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
            zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
            q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
            __syncthreads();

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                for (int i=0; i<Nq-block*BSZ; i++)
                {
                    dx = xq_sh[i] - x;
                    dy = yq_sh[i] - y;
                    dz = zq_sh[i] - z;
                    r  = sqrt(dx*dx + dy*dy + dz*dz);

                    sum += q_sh[i]/(E_1*r);
                }
            }

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                F[I] = sum;
            }
        }
    }

    __global__ void compute_RHS_dipole(REAL *F, REAL *xq, REAL *yq, REAL *zq,
                                REAL *px, REAL *py, REAL *pz, 
                                REAL *xi, REAL *yi, REAL *zi,
                                int *sizeTar, int Nq, REAL E_1, 
                                int NCRIT, int BpT)
    {
        int II = threadIdx.x + blockIdx.x*NCRIT;
        int I;
        REAL x, y, z, sum;
        REAL dx, dy, dz, r;
        __shared__ REAL xq_sh[BSZ], yq_sh[BSZ], zq_sh[BSZ];
        __shared__ REAL px_sh[BSZ], py_sh[BSZ], pz_sh[BSZ];

        for (int iblock=0; iblock<BpT; iblock++)
        {
            I = II + iblock*BSZ;
            x = xi[I];
            y = yi[I];
            z = zi[I];
            sum = 0.;

            for (int block=0; block<(Nq-1)/BSZ; block++)
            {
                __syncthreads();
                xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
                yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
                zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
                px_sh[threadIdx.x] = px[block*BSZ+threadIdx.x];
                py_sh[threadIdx.x] = py[block*BSZ+threadIdx.x];
                pz_sh[threadIdx.x] = pz[block*BSZ+threadIdx.x];
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int i=0; i<BSZ; i++)
                    {
                        dx = x - xq_sh[i];
                        dy = y - yq_sh[i];
                        dz = z - zq_sh[i];
                        r  = sqrt(dx*dx + dy*dy + dz*dz);

                        sum += (px_sh[i]*dx + py_sh[i]*dy + pz_sh[i]*dz)/(E_1*r*r*r);
                    }
                }
            }

            int block = (Nq-1)/BSZ; 
            __syncthreads();
            xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
            yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
            zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
            px_sh[threadIdx.x] = px[block*BSZ+threadIdx.x];
            py_sh[threadIdx.x] = py[block*BSZ+threadIdx.x];
            pz_sh[threadIdx.x] = pz[block*BSZ+threadIdx.x];
            __syncthreads();

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                for (int i=0; i<Nq-block*BSZ; i++)
                {
                    dx = x - xq_sh[i];
                    dy = y - yq_sh[i];
                    dz = z - zq_sh[i];
                    r  = sqrt(dx*dx + dy*dy + dz*dz);

                    sum += (px_sh[i]*dx + py_sh[i]*dy + pz_sh[i]*dz)/(E_1*r*r*r);
                }
            }

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                F[I] += sum;
            }
        }
    }
    __global__ void compute_RHS_quadrupole(REAL *F, REAL *xq, REAL *yq, REAL *zq,
                                            REAL *Qxx, REAL *Qxy, REAL *Qxz, 
                                            REAL *Qyx, REAL *Qyy, REAL *Qyz, 
                                            REAL *Qzx, REAL *Qzy, REAL *Qzz, 
                                            REAL *xi, REAL *yi, REAL *zi,
                                            int *sizeTar, int Nq, REAL E_1, 
                                            int NCRIT, int BpT)
    {
        int II = threadIdx.x + blockIdx.x*NCRIT;
        int I;
        REAL x, y, z, sum;
        REAL dx, dy, dz, r, r2;
        __shared__ REAL xq_sh[BSZ], yq_sh[BSZ], zq_sh[BSZ];
        __shared__ REAL Qxx_sh[BSZ], Qxy_sh[BSZ], Qxz_sh[BSZ];
        __shared__ REAL Qyx_sh[BSZ], Qyy_sh[BSZ], Qyz_sh[BSZ];
        __shared__ REAL Qzx_sh[BSZ], Qzy_sh[BSZ], Qzz_sh[BSZ];

        for (int iblock=0; iblock<BpT; iblock++)
        {
            I = II + iblock*BSZ;
            x = xi[I];
            y = yi[I];
            z = zi[I];
            sum = 0.;

            for (int block=0; block<(Nq-1)/BSZ; block++)
            {
                __syncthreads();
                xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
                yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
                zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
                Qxx_sh[threadIdx.x] = Qxx[block*BSZ+threadIdx.x];
                Qxy_sh[threadIdx.x] = Qxy[block*BSZ+threadIdx.x];
                Qxz_sh[threadIdx.x] = Qxz[block*BSZ+threadIdx.x];
                Qyx_sh[threadIdx.x] = Qyx[block*BSZ+threadIdx.x];
                Qyy_sh[threadIdx.x] = Qyy[block*BSZ+threadIdx.x];
                Qyz_sh[threadIdx.x] = Qyz[block*BSZ+threadIdx.x];
                Qzx_sh[threadIdx.x] = Qzx[block*BSZ+threadIdx.x];
                Qzy_sh[threadIdx.x] = Qzy[block*BSZ+threadIdx.x];
                Qzz_sh[threadIdx.x] = Qzz[block*BSZ+threadIdx.x];
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int i=0; i<BSZ; i++)
                    {
                        dx = x - xq_sh[i];
                        dy = y - yq_sh[i];
                        dz = z - zq_sh[i];
                        r2 = dx*dx + dy*dy + dz*dz;
                        r  = sqrt(r2);

                        sum += (Qxx_sh[i]*dx*dx + Qxy_sh[i]*dx*dy + Qxz_sh[i]*dx*dz 
                               +Qyx_sh[i]*dy*dx + Qyy_sh[i]*dy*dy + Qyz_sh[i]*dy*dz 
                               +Qzx_sh[i]*dz*dx + Qzy_sh[i]*dz*dy + Qzz_sh[i]*dz*dz)/(2*r2*r2*r*E_1);
                               
                    }
                }
            }

            int block = (Nq-1)/BSZ; 
            __syncthreads();
            xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
            yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
            zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
            Qxx_sh[threadIdx.x] = Qxx[block*BSZ+threadIdx.x];
            Qxy_sh[threadIdx.x] = Qxy[block*BSZ+threadIdx.x];
            Qxz_sh[threadIdx.x] = Qxz[block*BSZ+threadIdx.x];
            Qyx_sh[threadIdx.x] = Qyx[block*BSZ+threadIdx.x];
            Qyy_sh[threadIdx.x] = Qyy[block*BSZ+threadIdx.x];
            Qyz_sh[threadIdx.x] = Qyz[block*BSZ+threadIdx.x];
            Qzx_sh[threadIdx.x] = Qzx[block*BSZ+threadIdx.x];
            Qzy_sh[threadIdx.x] = Qzy[block*BSZ+threadIdx.x];
            Qzz_sh[threadIdx.x] = Qzz[block*BSZ+threadIdx.x];
            __syncthreads();

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                for (int i=0; i<Nq-block*BSZ; i++)
                {
                    dx = x - xq_sh[i];
                    dy = y - yq_sh[i];
                    dz = z - zq_sh[i];
                    r2 = dx*dx + dy*dy + dz*dz;
                    r  = sqrt(r2);

                    sum += (Qxx_sh[i]*dx*dx + Qxy_sh[i]*dx*dy + Qxz_sh[i]*dx*dz 
                           +Qyx_sh[i]*dy*dx + Qyy_sh[i]*dy*dy + Qyz_sh[i]*dy*dz 
                           +Qzx_sh[i]*dz*dx + Qzy_sh[i]*dz*dy + Qzz_sh[i]*dz*dz)/(2*r2*r2*r*E_1);
                }
            }

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                F[I] += sum;
            }
        }
    }

    __global__ void compute_RHSKt(REAL *Fx, REAL *Fy, REAL *Fz, REAL *xq, REAL *yq, REAL *zq,
                                REAL *q, REAL *xi, REAL *yi, REAL *zi,
                                int *sizeTar, int Nq, REAL E_1, 
                                int NCRIT, int BpT)
    {
        int II = threadIdx.x + blockIdx.x*NCRIT;
        int I;
        REAL x, y, z, sum_x, sum_y, sum_z;
        REAL dx, dy, dz, r, aux;
        __shared__ REAL xq_sh[BSZ], yq_sh[BSZ], zq_sh[BSZ], q_sh[BSZ];

        for (int iblock=0; iblock<BpT; iblock++)
        {
            I = II + iblock*BSZ;
            x = xi[I];
            y = yi[I];
            z = zi[I];
            sum_x = 0., sum_y = 0, sum_z = 0;

            for (int block=0; block<(Nq-1)/BSZ; block++)
            {
                __syncthreads();
                xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
                yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
                zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
                q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int i=0; i<BSZ; i++)
                    {
                        dx = x - xq_sh[i];
                        dy = y - yq_sh[i];
                        dz = z - zq_sh[i];
                        r  = sqrt(dx*dx + dy*dy + dz*dz);
                        aux = -q_sh[i]/(r*r*r);

                        sum_x += aux*dx;
                        sum_y += aux*dy;
                        sum_z += aux*dz;
                    }
                }
            }

            int block = (Nq-1)/BSZ; 
            __syncthreads();
            xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
            yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
            zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
            q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
            __syncthreads();

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                for (int i=0; i<Nq-block*BSZ; i++)
                {
                    dx = x - xq_sh[i];
                    dy = y - yq_sh[i];
                    dz = z - zq_sh[i];
                    r  = sqrt(dx*dx + dy*dy + dz*dz);
                    aux = -q_sh[i]/(r*r*r);

                    sum_x += aux*dx;
                    sum_y += aux*dy;
                    sum_z += aux*dz;
                }
            }

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                Fx[I] = sum_x;
                Fy[I] = sum_y;
                Fz[I] = sum_z;
            }
        }
    }

    __global__ void coulomb_direct(REAL *xq, REAL *yq, REAL *zq,
                                REAL *q, REAL *point_energy, int Nq)
    {
        int I = threadIdx.x + blockIdx.x*blockDim.x;
        REAL x, y, z, sum;
        REAL dx, dy, dz, r, eps = 1e-16;
        __shared__ REAL xq_sh[BSZ], yq_sh[BSZ], zq_sh[BSZ], q_sh[BSZ];

        x = xq[I];
        y = yq[I];
        z = zq[I];
        sum = 0.;

        for (int block=0; block<(Nq-1)/BSZ; block++)
        {
            __syncthreads();
            xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
            yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
            zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
            q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
            __syncthreads();

            if (I<Nq)
            {
                for (int j=0; j<BSZ; j++)
                {
                    dx = x - xq_sh[j];
                    dy = y - yq_sh[j];
                    dz = z - zq_sh[j];
                    r  = rsqrt(dx*dx + dy*dy + dz*dz + eps*eps);

                    if (r<1e12)
                        sum += q_sh[j]*r;
                }
            }
        }

        int block = (Nq-1)/BSZ; 
        __syncthreads();
        xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
        yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
        zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
        q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
        __syncthreads();

        if (I<Nq)
        {
            for (int j=0; j<Nq-block*BSZ; j++)
            {
                dx = x - xq_sh[j];
                dy = y - yq_sh[j];
                dz = z - zq_sh[j];
                r  = rsqrt(dx*dx + dy*dy + dz*dz + eps*eps);

                if (r<1e12)
                    sum += q_sh[j]*r;
            }
        }

        if (I<Nq)
        {
            point_energy[I] = q[I]*sum;
        }
    }


    __device__ void coulomb_phi_multipole_block(REAL x, REAL y, REAL z, REAL *xq, REAL *yq, REAL *zq, REAL *q,
                                      REAL *px, REAL *py, REAL *pz, REAL *Qxx, REAL *Qxy, REAL *Qxz,
                                      REAL *Qyx, REAL *Qyy, REAL *Qyz, REAL *Qzx, REAL *Qzy, REAL *Qzz,
                                      REAL &phi_coul, int size)

    {
        REAL r, r3, r5, sum = 0;
        REAL eps = 1e-15;
        REAL T0, T1[3], T2[3][3], Ri[3];

        for (int j=0; j<size; j++)
        {
            Ri[0] = x - xq[j];
            Ri[1] = y - yq[j];
            Ri[2] = z - zq[j];
            r  = rsqrt(Ri[0]*Ri[0] + Ri[1]*Ri[1] + Ri[2]*Ri[2] + eps*eps);
            r3 = r*r*r;
            r5 = r3*r*r;

            if (r<1e12)
            {
                T0 = r;
                for (int k=0; k<3; k++)
                {
                    T1[k] = Ri[k]*r3;
                    for (int l=0; l<3; l++)
                    {
                        T2[k][l] = Ri[k]*Ri[l]*r5; 
                    }

                }

                sum += T0*q[j] + T1[0]*px[j] + T1[1]*py[j] + T1[2]*pz[j]
                    + 0.5*(T2[0][0]*Qxx[j] + T2[0][1]*Qxy[j] + T2[0][2]*Qxz[j]
                         + T2[1][0]*Qyx[j] + T2[1][1]*Qyy[j] + T2[1][2]*Qyz[j]
                         + T2[2][0]*Qzx[j] + T2[2][1]*Qzy[j] + T2[2][2]*Qzz[j]);
            }
        }
        phi_coul += sum;
    }

    __device__ void coulomb_dphi_multipole_block(REAL x, REAL y, REAL z, REAL *xq, REAL *yq, REAL *zq, REAL *q,
                                      REAL *px, REAL *py, REAL *pz, REAL *Qxx, REAL *Qxy, REAL *Qxz,
                                      REAL *Qyx, REAL *Qyy, REAL *Qyz, REAL *Qzx, REAL *Qzy, REAL *Qzz, 
                                      REAL *thole, REAL thole_local, REAL *alpha, REAL alpha_local, 
									  int *polar_group, int local_polar_group, REAL &dphix_coul, 
                                      REAL &dphiy_coul, REAL &dphiz_coul, int size)
    {
        REAL r, r3, r5, r7;
        REAL eps = 1e-15;
        REAL T0, T1[3], T2[3][3], Ri[3], sum[3];
        REAL dkl, dkm;
        bool not_same_group;
        REAL scale3 = 1.0;
        REAL scale5 = 1.0;
        REAL scale7 = 1.0;
        REAL damp, gamma, expdamp;

        sum[0] = 0;
        sum[1] = 0;
        sum[2] = 0;
        for (int j=0; j<size; j++)
        {
            Ri[0] = x - xq[j];
            Ri[1] = y - yq[j];
            Ri[2] = z - zq[j];
            r  = rsqrt(Ri[0]*Ri[0] + Ri[1]*Ri[1] + Ri[2]*Ri[2] + eps*eps);
            r3 = r*r*r;
            r5 = r3*r*r;
            r7 = r5*r*r;

            if (local_polar_group==-1)
            {
                not_same_group = true;
            }
            else
            {
                gamma = min(thole_local, thole[j]);
                damp = pow(alpha_local*alpha[j],0.16666667);
                damp += 1e-12;
                damp = -gamma*(1/(r3*damp*damp*damp));
                expdamp = exp(damp);
                scale3 = 1 - expdamp;
                scale5 = 1 - expdamp*(1-damp);
                scale7 = 1 - expdamp*(1-damp+0.6*damp*damp);
                not_same_group = (local_polar_group!=polar_group[j]);
            }
                

            if ((r<1e12) && not_same_group)
            {
                for (int k=0; k<3; k++)
                {
                    T0 = -Ri[k]*r3* scale3;
                    for (int l=0; l<3; l++)
                    {
                        dkl = (REAL)(k==l);
                        T1[l] = dkl*r3*scale3 - 3*Ri[k]*Ri[l]*r5*scale5;

                        for (int m=0; m<3; m++)
                        {
                            dkm = (REAL)(k==m);
                            T2[l][m] = (dkm*Ri[l]+dkl*Ri[m])*r5*scale5 - 5*Ri[l]*Ri[m]*Ri[k]*r7*scale7;
                        }
                    }

                    sum[k] += T0*q[j] + T1[0]*px[j] + T1[1]*py[j] + T1[2]*pz[j]
                           + 0.5*(T2[0][0]*Qxx[j] + T2[0][1]*Qxy[j] + T2[0][2]*Qxz[j] 
                                + T2[1][0]*Qyx[j] + T2[1][1]*Qyy[j] + T2[1][2]*Qyz[j]
                                + T2[2][0]*Qzx[j] + T2[2][1]*Qzy[j] + T2[2][2]*Qzz[j]);
                }
            }
        }
        dphix_coul += sum[0];
        dphiy_coul += sum[1];
        dphiz_coul += sum[2];
    }    

    __device__ void coulomb_phi_multipole_Thole_block(REAL x, REAL y, REAL z, REAL *xq, REAL *yq, REAL *zq, 
												REAL *px, REAL *py, REAL *pz, REAL *alpha, REAL alpha_local, 
												REAL *thole, REAL thole_local, 
                                                int *connections_12, int start_12, int stop_12, REAL p12scale,
                                                int *connections_13, int start_13, int stop_13, REAL p13scale,
                                                REAL &phi, int j_start, int size)
	{
		REAL r, r3;
		REAL eps = 1e-15;
		REAL T1[3], Ri[3], sum; 
		REAL scale3 = 1.0; 
		REAL damp, gamma, expdamp;
        REAL pscale;


		sum = 0.0; 
		for (int j=0; j<size; j++) 
		{

            pscale = 1.0;
            for(int ii=start_12; ii<stop_12; ii++)
            {
                if (connections_12[ii]==(j_start+j))
                {
                    pscale = p12scale;
                }
            }

            for(int ii=start_13; ii<stop_13; ii++)
            {
                if (connections_13[ii]==(j_start+j))
                {
                    pscale = p13scale;
                }
            }

			Ri[0] = x - xq[j]; 
			Ri[1] = y - yq[j]; 
			Ri[2] = z - zq[j]; 

			r  = rsqrt(Ri[0]*Ri[0] + Ri[1]*Ri[1] + Ri[2]*Ri[2] + eps*eps);
			r3 = r*r*r;
	 
			gamma = min(thole_local, thole[j]);
			damp = pow(alpha_local*alpha[j],0.16666667);
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
		phi += sum; 

	} 

    __device__ void coulomb_dphi_multipole_Thole_block(REAL x, REAL y, REAL z, REAL *xq, REAL *yq, REAL *zq, 
                                                    REAL *px, REAL *py, REAL *pz, REAL *alpha, REAL alpha_local,
													REAL *thole, REAL thole_local,
                                                    int *connections_12, int start_12, int stop_12, REAL p12scale,
                                                    int *connections_13, int start_13, int stop_13, REAL p13scale,
                                                    REAL &dphix_coul, REAL &dphiy_coul, REAL &dphiz_coul, int j_start, int size)
    {
        REAL r, r3, r5;
        REAL eps = 1e-15;
        REAL T1[3], Ri[3], sum[3];
        REAL dkl, dkm;
		REAL scale3 = 1.0; 
		REAL scale5 = 1.0; 
		REAL damp, gamma, expdamp;
        REAL pscale;

        sum[0] = 0;
        sum[1] = 0;
        sum[2] = 0;
        for (int j=0; j<size; j++)
        {
            pscale = 1.0;
            for(int ii=start_12; ii<stop_12; ii++)
            {
                if (connections_12[ii]==(j_start+j))
                {
                    pscale = p12scale;
                }
            }

            for(int ii=start_13; ii<stop_13; ii++)
            {
                if (connections_13[ii]==(j_start+j))
                {
                    pscale = p13scale;
                }
            }

            Ri[0] = x - xq[j];
            Ri[1] = y - yq[j];
            Ri[2] = z - zq[j];
            r  = rsqrt(Ri[0]*Ri[0] + Ri[1]*Ri[1] + Ri[2]*Ri[2] + eps*eps);
            r3 = r*r*r;
            r5 = r3*r*r;

            gamma = min(thole_local, thole[j]);
            damp = pow(alpha_local*alpha[j],0.16666667);
            damp += 1e-12;
            damp = -gamma*(1/(r3*damp*damp*damp));

            expdamp = exp(damp);
            scale3 = 1 - expdamp;
            scale5 = 1 - expdamp*(1-damp);

            if (r<1e12)
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
        dphix_coul += sum[0];
        dphiy_coul += sum[1];
        dphiz_coul += sum[2];
    }

    __device__ void coulomb_ddphi_multipole_Thole_block(REAL x, REAL y, REAL z, REAL *xq, REAL *yq, REAL *zq, 
                                                    REAL *px, REAL *py, REAL *pz, REAL *alpha, REAL alpha_local,
													REAL *thole, REAL thole_local,
                                                    int *connections_12, int start_12, int stop_12, REAL p12scale,
                                                    int *connections_13, int start_13, int stop_13, REAL p13scale,
												    REAL &ddphixx_coul, REAL &ddphixy_coul, REAL &ddphixz_coul,
												    REAL &ddphiyx_coul, REAL &ddphiyy_coul, REAL &ddphiyz_coul,
												    REAL &ddphizx_coul, REAL &ddphizy_coul, REAL &ddphizz_coul, 
                                                    int j_start, int size)
	{
		REAL r, r3, r5, r7;
		REAL eps = 1e-15;
		REAL T1[3], Ri[3], sum[3][3];
		REAL dkl, dkm, dlm;
		REAL scale5 = 1.0;
		REAL scale7 = 1.0;
		REAL damp, gamma, expdamp;
        REAL pscale;

		sum[0][0] = 0.0;
		sum[0][1] = 0.0;
		sum[0][2] = 0.0;
		sum[1][0] = 0.0;
		sum[1][1] = 0.0;
		sum[1][2] = 0.0;
		sum[2][0] = 0.0;
		sum[2][1] = 0.0;
		sum[2][2] = 0.0;

		for (int j=0; j<size; j++)
		{
            pscale = 1.0;
            for(int ii=start_12; ii<stop_12; ii++)
            {
                if (connections_12[ii]==(j_start+j))
                {
                    pscale = p12scale;
                }
            }

            for(int ii=start_13; ii<stop_13; ii++)
            {
                if (connections_13[ii]==(j_start+j))
                {
                    pscale = p13scale;
                }
            }
            
            Ri[0] = x - xq[j];
			Ri[1] = y - yq[j];
			Ri[2] = z - zq[j];

			r  = rsqrt(Ri[0]*Ri[0] + Ri[1]*Ri[1] + Ri[2]*Ri[2] + eps*eps);
			r3 = r*r*r;
			r5 = r3*r*r;
			r7 = r5*r*r;

			gamma = min(thole_local, thole[j]);
			damp = pow(alpha_local*alpha[j],0.16666667);
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
							T1[m] = -3*(dkm*Ri[l]+dkl*Ri[m]+dlm*Ri[k])*r5*scale5*pscale + 15*Ri[l]*Ri[m]*Ri[k]*r7*scale7*pscale;

						}
						sum[k][l] += T1[0]*px[j] + T1[1]*py[j] + T1[2]*pz[j];
					}
				}
			}
		}


        ddphixx_coul += sum[0][0];
        ddphixy_coul += sum[0][1];
        ddphixz_coul += sum[0][2];
        ddphiyx_coul += sum[1][0];
        ddphiyy_coul += sum[1][1];
        ddphiyz_coul += sum[1][2];
        ddphizx_coul += sum[2][0];
        ddphizy_coul += sum[2][1];
        ddphizz_coul += sum[2][2];
	}

    __device__ void coulomb_ddphi_multipole_block(REAL x, REAL y, REAL z, REAL *xq, REAL *yq, REAL *zq, REAL *q,
                                      REAL *px, REAL *py, REAL *pz, REAL *Qxx, REAL *Qxy, REAL *Qxz,
                                      REAL *Qyx, REAL *Qyy, REAL *Qyz, REAL *Qzx, REAL *Qzy, REAL *Qzz,
                                      REAL &ddphixx_coul, REAL &ddphixy_coul, REAL &ddphixz_coul,
                                      REAL &ddphiyx_coul, REAL &ddphiyy_coul, REAL &ddphiyz_coul,
                                      REAL &ddphizx_coul, REAL &ddphizy_coul, REAL &ddphizz_coul, int size)
    {
        REAL r, r3, r5, r7, r9;
        REAL eps = 1e-15;
        REAL T0, T1[3], T2[3][3], Ri[3], sum[3][3];
        REAL dkl, dkm, dlm, dkn, dln;

        sum[0][0] = 0.0;
        sum[0][1] = 0.0;
        sum[0][2] = 0.0;
        sum[1][0] = 0.0;
        sum[1][1] = 0.0;
        sum[1][2] = 0.0;
        sum[2][0] = 0.0;
        sum[2][1] = 0.0;
        sum[2][2] = 0.0;

        for (int j=0; j<size; j++)
        {
            Ri[0] = x - xq[j];
            Ri[1] = y - yq[j];
            Ri[2] = z - zq[j];
            r  = rsqrt(Ri[0]*Ri[0] + Ri[1]*Ri[1] + Ri[2]*Ri[2] + eps*eps);
            r3 = r*r*r;
            r5 = r3*r*r;
            r7 = r5*r*r;
            r9 = r3*r3*r3;

            if (r<1e12)
            {
                for (int k=0; k<3; k++)
                {
                    for (int l=0; l<3; l++)
                    {
                        dkl = (REAL)(k==l);
                        T0 = -dkl*r3 + + 3*Ri[k]*Ri[l]*r5;

                        for (int m=0; m<3; m++)
                        {
                            dkm = (REAL)(k==m);
                            dlm = (REAL)(l==m);
                            T1[m] = -3*(dkm*Ri[l]+dkl*Ri[m]+dlm*Ri[k])*r5 + 15*Ri[l]*Ri[m]*Ri[k]*r7;

                            for (int n=0; n<3; n++)
                            {
                                dkn = (REAL)(k==n);
                                dln = (REAL)(l==n);
                                T2[m][n] = 35*Ri[k]*Ri[l]*Ri[m]*Ri[n]*r9
                                        - 5*(Ri[m]*Ri[n]*dkl + Ri[l]*Ri[n]*dkm
                                           + Ri[m]*Ri[l]*dkn + Ri[k]*Ri[n]*dlm
                                           + Ri[m]*Ri[k]*dln)*r7
                                           + (dkm*dln + dlm*dkn)*r5; 
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

        ddphixx_coul += sum[0][0];
        ddphixy_coul += sum[0][1];
        ddphixz_coul += sum[0][2];
        ddphiyx_coul += sum[1][0];
        ddphiyy_coul += sum[1][1];
        ddphiyz_coul += sum[1][2];
        ddphizx_coul += sum[2][0];
        ddphizy_coul += sum[2][1];
        ddphizz_coul += sum[2][2];
    }

    __global__ void compute_induced_dipole(REAL *xq, REAL *yq, REAL *zq,
                                        REAL *q, REAL *px, REAL *py, REAL *pz,
                                        REAL *px_pol, REAL *py_pol, REAL *pz_pol,
                                        REAL *Qxx, REAL *Qxy, REAL *Qxz,
                                        REAL *Qyx, REAL *Qyy, REAL *Qyz,
                                        REAL *Qzx, REAL *Qzy, REAL *Qzz,
                                        REAL *alphaxx, REAL *alphaxy, REAL *alphaxz,
                                        REAL *alphayx, REAL *alphayy, REAL *alphayz,
                                        REAL *alphazx, REAL *alphazy, REAL *alphazz, REAL *thole, int *polar_group,
                                        int *connections_12, int* pointer_connections_12,
                                        int *connections_13, int* pointer_connections_13,
                                        REAL *dphix_reac, REAL *dphiy_reac, REAL *dphiz_reac,
                                        REAL E, int Nq)
    {
        int I = threadIdx.x + blockIdx.x*blockDim.x;
        REAL x, y, z, alpha_local;
        REAL dphix_coul = 0, dphiy_coul = 0, dphiz_coul = 0;
        REAL dx, dy, dz, r, eps = 1e-16;
        __shared__ REAL xq_sh[BSZ], yq_sh[BSZ], zq_sh[BSZ], q_sh[BSZ];
        __shared__ REAL px_sh[BSZ], py_sh[BSZ], pz_sh[BSZ];
        __shared__ REAL px_pol_sh[BSZ], py_pol_sh[BSZ], pz_pol_sh[BSZ];
        __shared__ REAL Qxx_sh[BSZ], Qxy_sh[BSZ], Qxz_sh[BSZ];
        __shared__ REAL Qyx_sh[BSZ], Qyy_sh[BSZ], Qyz_sh[BSZ];
        __shared__ REAL Qzx_sh[BSZ], Qzy_sh[BSZ], Qzz_sh[BSZ];
        __shared__ REAL alpha_sh[BSZ], thole_sh[BSZ];
        __shared__ int polar_group_sh[BSZ];

        x = xq[I];
        y = yq[I];
        z = zq[I];
        alpha_local = alphaxx[I]; // Using alphaxx because it is usually a scalar (not tensor)
		REAL thole_local = thole[I];

        int local_polar_group = polar_group[I];
        int ptr_conn12_start = pointer_connections_12[I];
        int ptr_conn12_end = pointer_connections_12[I+1];
        int ptr_conn13_start = pointer_connections_13[I];
        int ptr_conn13_end = pointer_connections_13[I+1];

        for (int block=0; block<(Nq-1)/BSZ; block++)
        {
            __syncthreads();
            xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
            yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
            zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
            q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
            px_sh[threadIdx.x] = px[block*BSZ+threadIdx.x];
            py_sh[threadIdx.x] = py[block*BSZ+threadIdx.x];
            pz_sh[threadIdx.x] = pz[block*BSZ+threadIdx.x];
            px_pol_sh[threadIdx.x] = px_pol[block*BSZ+threadIdx.x];
            py_pol_sh[threadIdx.x] = py_pol[block*BSZ+threadIdx.x];
            pz_pol_sh[threadIdx.x] = pz_pol[block*BSZ+threadIdx.x];
            Qxx_sh[threadIdx.x] = Qxx[block*BSZ+threadIdx.x];
            Qxy_sh[threadIdx.x] = Qxy[block*BSZ+threadIdx.x];
            Qxz_sh[threadIdx.x] = Qxz[block*BSZ+threadIdx.x];
            Qyx_sh[threadIdx.x] = Qyx[block*BSZ+threadIdx.x];
            Qyy_sh[threadIdx.x] = Qyy[block*BSZ+threadIdx.x];
            Qyz_sh[threadIdx.x] = Qyz[block*BSZ+threadIdx.x];
            Qzx_sh[threadIdx.x] = Qzx[block*BSZ+threadIdx.x];
            Qzy_sh[threadIdx.x] = Qzy[block*BSZ+threadIdx.x];
            Qzz_sh[threadIdx.x] = Qzz[block*BSZ+threadIdx.x];
            alpha_sh[threadIdx.x] = alphaxx[block*BSZ+threadIdx.x]; // using alphaxx as it usually is a scalar
            thole_sh[threadIdx.x] = thole[block*BSZ+threadIdx.x];
            polar_group_sh[threadIdx.x] = polar_group[block*BSZ+threadIdx.x];
            __syncthreads();

            if (I<Nq)
            {
                // Polarization due to permanent multipoles only
                coulomb_dphi_multipole_block(x, y, z, xq_sh, yq_sh, zq_sh, q_sh,
                                        px_sh, py_sh, pz_sh, Qxx_sh, Qxy_sh, Qxz_sh,
                                        Qyx_sh, Qyy_sh, Qyz_sh, Qzx_sh, Qzy_sh, Qzz_sh, thole_sh, thole_local,
                                        alpha_sh, alpha_local, polar_group_sh, local_polar_group, dphix_coul, dphiy_coul, 
                                        dphiz_coul, BSZ);

                // Polarization due to induced dipoles only
                coulomb_dphi_multipole_Thole_block(x, y, z, xq_sh, yq_sh, zq_sh, 
                                                px_pol_sh, py_pol_sh, pz_pol_sh, 
                                                alpha_sh, alpha_local, thole_sh, thole_local,
                                                connections_12, ptr_conn12_start, ptr_conn12_end, 1.0,
                                                connections_13, ptr_conn13_start, ptr_conn13_end, 1.0,
                                                dphix_coul, dphiy_coul, dphiz_coul, block*BSZ, BSZ);
            }
        }

        int block = (Nq-1)/BSZ; 
        __syncthreads();
        xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
        yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
        zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
        q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
        px_sh[threadIdx.x] = px[block*BSZ+threadIdx.x];
        py_sh[threadIdx.x] = py[block*BSZ+threadIdx.x];
        pz_sh[threadIdx.x] = pz[block*BSZ+threadIdx.x];
        px_pol_sh[threadIdx.x] = px_pol[block*BSZ+threadIdx.x];
        py_pol_sh[threadIdx.x] = py_pol[block*BSZ+threadIdx.x];
        pz_pol_sh[threadIdx.x] = pz_pol[block*BSZ+threadIdx.x];
        Qxx_sh[threadIdx.x] = Qxx[block*BSZ+threadIdx.x];
        Qxy_sh[threadIdx.x] = Qxy[block*BSZ+threadIdx.x];
        Qxz_sh[threadIdx.x] = Qxz[block*BSZ+threadIdx.x];
        Qyx_sh[threadIdx.x] = Qyx[block*BSZ+threadIdx.x];
        Qyy_sh[threadIdx.x] = Qyy[block*BSZ+threadIdx.x];
        Qyz_sh[threadIdx.x] = Qyz[block*BSZ+threadIdx.x];
        Qzx_sh[threadIdx.x] = Qzx[block*BSZ+threadIdx.x];
        Qzy_sh[threadIdx.x] = Qzy[block*BSZ+threadIdx.x];
        Qzz_sh[threadIdx.x] = Qzz[block*BSZ+threadIdx.x];
        alpha_sh[threadIdx.x] = alphaxx[block*BSZ+threadIdx.x];// using alphaxx as it usually is a scalar
        thole_sh[threadIdx.x] = thole[block*BSZ+threadIdx.x];
        polar_group_sh[threadIdx.x] = polar_group[block*BSZ+threadIdx.x];
        __syncthreads();

        if (I<Nq)
        {
            // Polarization due to permanent multipoles only
            coulomb_dphi_multipole_block(x, y, z, xq_sh, yq_sh, zq_sh, q_sh,
                                    px_sh, py_sh, pz_sh, Qxx_sh, Qxy_sh, Qxz_sh,
                                    Qyx_sh, Qyy_sh, Qyz_sh, Qzx_sh, Qzy_sh, Qzz_sh, thole_sh, thole_local,
                                    alpha_sh, alpha_local, polar_group_sh, local_polar_group, dphix_coul, dphiy_coul, 
                                    dphiz_coul, (Nq-block*BSZ));

            // Polarization due to induced dipoles only
            coulomb_dphi_multipole_Thole_block(x, y, z, xq_sh, yq_sh, zq_sh, 
                                            px_pol_sh, py_pol_sh, pz_pol_sh, 
                                            alpha_sh, alpha_local, thole_sh, thole_local,
                                            connections_12, ptr_conn12_start, ptr_conn12_end, 1.0,
                                            connections_13, ptr_conn13_start, ptr_conn13_end, 1.0,
                                            dphix_coul, dphiy_coul, dphiz_coul, block*BSZ, (Nq-block*BSZ));
        }

        __syncthreads();
		REAL SOR = 0.7;
        if (I<Nq)
        {
            px_pol[I] = px_pol[I]*(1-SOR) + 
					    (-alphaxx[I]*(dphix_coul/(E)+4*M_PI*dphix_reac[I]) 
                         -alphaxy[I]*(dphiy_coul/(E)+4*M_PI*dphiy_reac[I]) 
                         -alphaxz[I]*(dphiz_coul/(E)+4*M_PI*dphiz_reac[I]))*SOR;
            py_pol[I] = py_pol[I]*(1-SOR) + 
						(-alphayx[I]*(dphix_coul/(E)+4*M_PI*dphix_reac[I]) 
                         -alphayy[I]*(dphiy_coul/(E)+4*M_PI*dphiy_reac[I]) 
                         -alphayz[I]*(dphiz_coul/(E)+4*M_PI*dphiz_reac[I]))*SOR;
            pz_pol[I] = pz_pol[I]*(1-SOR) + 
						(-alphazx[I]*(dphix_coul/(E)+4*M_PI*dphix_reac[I]) 
                         -alphazy[I]*(dphiy_coul/(E)+4*M_PI*dphiy_reac[I]) 
                         -alphazz[I]*(dphiz_coul/(E)+4*M_PI*dphiz_reac[I]))*SOR;

        }
    }
    
    __global__ void coulomb_energy_multipole(REAL *xq, REAL *yq, REAL *zq, REAL *q, 
                                             REAL *px, REAL *py, REAL *pz, 
                                             REAL *px_pol, REAL *py_pol, REAL *pz_pol, 
                                             REAL *Qxx, REAL *Qxy, REAL *Qxz, 
                                             REAL *Qyx, REAL *Qyy, REAL *Qyz, 
                                             REAL *Qzx, REAL *Qzy, REAL *Qzz, 
                                             REAL *alphaxx, REAL *thole, 
                                             int *connections_12, int* pointer_connections_12,
                                             int *connections_13, int* pointer_connections_13,
                                             REAL p12scale, REAL p13scale, REAL *point_energy, int Nq)
    {
        int I = threadIdx.x + blockIdx.x*blockDim.x;
        REAL x, y, z, phi_coul = 0;
        REAL dphix_coul = 0, dphiy_coul = 0, dphiz_coul = 0;
        REAL ddphixx_coul = 0, ddphixy_coul = 0, ddphixz_coul = 0;
        REAL ddphiyx_coul = 0, ddphiyy_coul = 0, ddphiyz_coul = 0;
        REAL ddphizx_coul = 0, ddphizy_coul = 0, ddphizz_coul = 0;
        REAL dx, dy, dz, r, eps = 1e-16, alpha_local, thole_local;
        __shared__ REAL xq_sh[BSZ], yq_sh[BSZ], zq_sh[BSZ], q_sh[BSZ];
        __shared__ REAL px_sh[BSZ], py_sh[BSZ], pz_sh[BSZ];
        __shared__ REAL px_pol_sh[BSZ], py_pol_sh[BSZ], pz_pol_sh[BSZ];
        __shared__ REAL Qxx_sh[BSZ], Qxy_sh[BSZ], Qxz_sh[BSZ];
        __shared__ REAL Qyx_sh[BSZ], Qyy_sh[BSZ], Qyz_sh[BSZ];
        __shared__ REAL Qzx_sh[BSZ], Qzy_sh[BSZ], Qzz_sh[BSZ];
        __shared__ REAL alpha_sh[BSZ], thole_sh[BSZ];
        __shared__ int dummy[BSZ];
		REAL *dummy2;

        x = xq[I];
        y = yq[I];
        z = zq[I];

        int ptr_conn12_start = pointer_connections_12[I];
        int ptr_conn12_end = pointer_connections_12[I+1];
        int ptr_conn13_start = pointer_connections_13[I];
        int ptr_conn13_end = pointer_connections_13[I+1];

        alpha_local = alphaxx[I]; // Using alphaxx because it is usually a scalar (not tensor)
		thole_local = thole[I];

        for (int block=0; block<(Nq-1)/BSZ; block++)
        {
            __syncthreads();
            xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
            yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
            zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
            q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
            px_sh[threadIdx.x] = px[block*BSZ+threadIdx.x];
            py_sh[threadIdx.x] = py[block*BSZ+threadIdx.x];
            pz_sh[threadIdx.x] = pz[block*BSZ+threadIdx.x];
            px_pol_sh[threadIdx.x] = px_pol[block*BSZ+threadIdx.x];
            py_pol_sh[threadIdx.x] = py_pol[block*BSZ+threadIdx.x];
            pz_pol_sh[threadIdx.x] = pz_pol[block*BSZ+threadIdx.x];
            Qxx_sh[threadIdx.x] = Qxx[block*BSZ+threadIdx.x];
            Qxy_sh[threadIdx.x] = Qxy[block*BSZ+threadIdx.x];
            Qxz_sh[threadIdx.x] = Qxz[block*BSZ+threadIdx.x];
            Qyx_sh[threadIdx.x] = Qyx[block*BSZ+threadIdx.x];
            Qyy_sh[threadIdx.x] = Qyy[block*BSZ+threadIdx.x];
            Qyz_sh[threadIdx.x] = Qyz[block*BSZ+threadIdx.x];
            Qzx_sh[threadIdx.x] = Qzx[block*BSZ+threadIdx.x];
            Qzy_sh[threadIdx.x] = Qzy[block*BSZ+threadIdx.x];
            Qzz_sh[threadIdx.x] = Qzz[block*BSZ+threadIdx.x];
            alpha_sh[threadIdx.x] = alphaxx[block*BSZ+threadIdx.x]; // using alphaxx as it usually is a scalar
            thole_sh[threadIdx.x] = thole[block*BSZ+threadIdx.x];
            __syncthreads();

            if (I<Nq)
            {
                // Permanent multipoles
                coulomb_phi_multipole_block(x, y, z, xq_sh, yq_sh, zq_sh, q_sh,
                                        px_sh, py_sh, pz_sh, Qxx_sh, Qxy_sh, Qxz_sh,
                                        Qyx_sh, Qyy_sh, Qyz_sh, Qzx_sh, Qzy_sh, Qzz_sh,
                                        phi_coul, BSZ);

                coulomb_dphi_multipole_block(x, y, z, xq_sh, yq_sh, zq_sh, q_sh,
                                        px_sh, py_sh, pz_sh, Qxx_sh, Qxy_sh, Qxz_sh,
                                        Qyx_sh, Qyy_sh, Qyz_sh, Qzx_sh, Qzy_sh, Qzz_sh, 
										dummy2, 1.0, dummy2, 1.0, dummy, -1,
                                        dphix_coul, dphiy_coul, dphiz_coul, BSZ);

                coulomb_ddphi_multipole_block(x, y, z, xq_sh, yq_sh, zq_sh, q_sh,
                                        px_sh, py_sh, pz_sh, Qxx_sh, Qxy_sh, Qxz_sh,
                                        Qyx_sh, Qyy_sh, Qyz_sh, Qzx_sh, Qzy_sh, Qzz_sh,
                                        ddphixx_coul, ddphixy_coul, ddphixz_coul, 
                                        ddphiyx_coul, ddphiyy_coul, ddphiyz_coul, 
                                        ddphizx_coul, ddphizy_coul, ddphizz_coul, BSZ);

                // Induced dipoles
                coulomb_phi_multipole_Thole_block(x, y, z, xq_sh, yq_sh, zq_sh, px_pol_sh, py_pol_sh, pz_pol_sh,
                                                  alpha_sh, alpha_local, thole_sh, thole_local, 
                                                  connections_12, ptr_conn12_start, ptr_conn12_end, p12scale,
                                                  connections_13, ptr_conn13_start, ptr_conn13_end, p13scale,
                                                  phi_coul, block*BSZ, BSZ);  

                coulomb_dphi_multipole_Thole_block(x, y, z, xq_sh, yq_sh, zq_sh, px_pol_sh, py_pol_sh, pz_pol_sh,
                                                  alpha_sh, alpha_local, thole_sh, thole_local, 
                                                  connections_12, ptr_conn12_start, ptr_conn12_end, p12scale,
                                                  connections_13, ptr_conn13_start, ptr_conn13_end, p13scale,
                                                  dphix_coul, dphiy_coul, dphiz_coul, block*BSZ, BSZ);

                coulomb_ddphi_multipole_Thole_block(x, y, z, xq_sh, yq_sh, zq_sh, px_pol_sh, py_pol_sh, pz_pol_sh,
                                                  alpha_sh, alpha_local, thole_sh, thole_local, 
                                                  connections_12, ptr_conn12_start, ptr_conn12_end, p12scale,
                                                  connections_13, ptr_conn13_start, ptr_conn13_end, p13scale,
                                                  ddphixx_coul, ddphixy_coul, ddphixz_coul, 
                                                  ddphiyx_coul, ddphiyy_coul, ddphiyz_coul, 
                                                  ddphizx_coul, ddphizy_coul, ddphizz_coul, block*BSZ, BSZ);  
            }
        }

        int block = (Nq-1)/BSZ; 
        __syncthreads();
        xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
        yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
        zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
        q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
        px_sh[threadIdx.x] = px[block*BSZ+threadIdx.x];
        py_sh[threadIdx.x] = py[block*BSZ+threadIdx.x];
        pz_sh[threadIdx.x] = pz[block*BSZ+threadIdx.x];
        px_pol_sh[threadIdx.x] = px_pol[block*BSZ+threadIdx.x];
        py_pol_sh[threadIdx.x] = py_pol[block*BSZ+threadIdx.x];
        pz_pol_sh[threadIdx.x] = pz_pol[block*BSZ+threadIdx.x];
        Qxx_sh[threadIdx.x] = Qxx[block*BSZ+threadIdx.x];
        Qxy_sh[threadIdx.x] = Qxy[block*BSZ+threadIdx.x];
        Qxz_sh[threadIdx.x] = Qxz[block*BSZ+threadIdx.x];
        Qyx_sh[threadIdx.x] = Qyx[block*BSZ+threadIdx.x];
        Qyy_sh[threadIdx.x] = Qyy[block*BSZ+threadIdx.x];
        Qyz_sh[threadIdx.x] = Qyz[block*BSZ+threadIdx.x];
        Qzx_sh[threadIdx.x] = Qzx[block*BSZ+threadIdx.x];
        Qzy_sh[threadIdx.x] = Qzy[block*BSZ+threadIdx.x];
        Qzz_sh[threadIdx.x] = Qzz[block*BSZ+threadIdx.x];
        alpha_sh[threadIdx.x] = alphaxx[block*BSZ+threadIdx.x]; // using alphaxx as it usually is a scalar
        thole_sh[threadIdx.x] = thole[block*BSZ+threadIdx.x];
        __syncthreads();

        if (I<Nq)
        {
            coulomb_phi_multipole_block(x, y, z, xq_sh, yq_sh, zq_sh, q_sh,
                                    px_sh, py_sh, pz_sh, Qxx_sh, Qxy_sh, Qxz_sh,
                                    Qyx_sh, Qyy_sh, Qyz_sh, Qzx_sh, Qzy_sh, Qzz_sh,
                                    phi_coul, (Nq-block*BSZ));

            coulomb_dphi_multipole_block(x, y, z, xq_sh, yq_sh, zq_sh, q_sh,
                                    px_sh, py_sh, pz_sh, Qxx_sh, Qxy_sh, Qxz_sh,
                                    Qyx_sh, Qyy_sh, Qyz_sh, Qzx_sh, Qzy_sh, Qzz_sh, 
									dummy2, 1.0, dummy2, 1.0, dummy, -1,
                                    dphix_coul, dphiy_coul, dphiz_coul, (Nq-block*BSZ));

            coulomb_ddphi_multipole_block(x, y, z, xq_sh, yq_sh, zq_sh, q_sh,
                                    px_sh, py_sh, pz_sh, Qxx_sh, Qxy_sh, Qxz_sh,
                                    Qyx_sh, Qyy_sh, Qyz_sh, Qzx_sh, Qzy_sh, Qzz_sh,
                                    ddphixx_coul, ddphixy_coul, ddphixz_coul, 
                                    ddphiyx_coul, ddphiyy_coul, ddphiyz_coul, 
                                    ddphizx_coul, ddphizy_coul, ddphizz_coul, (Nq-block*BSZ));

            // Induced dipoles
            coulomb_phi_multipole_Thole_block(x, y, z, xq_sh, yq_sh, zq_sh, px_pol_sh, py_pol_sh, pz_pol_sh,
                                              alpha_sh, alpha_local, thole_sh, thole_local, 
                                              connections_12, ptr_conn12_start, ptr_conn12_end, p12scale,
                                              connections_13, ptr_conn13_start, ptr_conn13_end, p13scale,
                                              phi_coul, block*BSZ, (Nq-block*BSZ));  

            coulomb_dphi_multipole_Thole_block(x, y, z, xq_sh, yq_sh, zq_sh, px_pol_sh, py_pol_sh, pz_pol_sh,
                                              alpha_sh, alpha_local, thole_sh, thole_local, 
                                              connections_12, ptr_conn12_start, ptr_conn12_end, p12scale,
                                              connections_13, ptr_conn13_start, ptr_conn13_end, p13scale,
                                              dphix_coul, dphiy_coul, dphiz_coul, block*BSZ, (Nq-block*BSZ));

            coulomb_ddphi_multipole_Thole_block(x, y, z, xq_sh, yq_sh, zq_sh, px_pol_sh, py_pol_sh, pz_pol_sh,
                                              alpha_sh, alpha_local, thole_sh, thole_local, 
                                              connections_12, ptr_conn12_start, ptr_conn12_end, p12scale,
                                              connections_13, ptr_conn13_start, ptr_conn13_end, p13scale,
                                              ddphixx_coul, ddphixy_coul, ddphixz_coul, 
                                              ddphiyx_coul, ddphiyy_coul, ddphiyz_coul, 
                                              ddphizx_coul, ddphizy_coul, ddphizz_coul, block*BSZ, (Nq-block*BSZ));  

        }

        if (I<Nq)
        {
            point_energy[I] = (q[I]*phi_coul
                      + px[I]*dphix_coul + py[I]*dphiy_coul + pz[I]*dphiz_coul
                      +(Qxx[I]*ddphixx_coul + Qxy[I]*ddphixy_coul + Qxz[I]*ddphixz_coul
                      + Qxy[I]*ddphiyx_coul + Qyy[I]*ddphiyy_coul + Qyz[I]*ddphiyz_coul
                      + Qxz[I]*ddphizx_coul + Qzy[I]*ddphizy_coul + Qzz[I]*ddphizz_coul)/6.);
                        // Energy calculated with p_tot-p_pol (rather than p_tot) to account for polarization energy
        }


    }

    """%{'blocksize':BSZ, 'Nmult':Nm, 'K_near':K_fine, 'Ptree':P, 'precision':REAL}, nvcc="nvcc", options=["-use_fast_math","-Xptxas=-v"])#,-abi=no"])

    return mod
