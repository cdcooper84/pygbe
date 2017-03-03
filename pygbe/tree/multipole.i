%module multipole

%{
#define SWIG_FILE_WITH_INIT
extern void multipole_c(double *K, int KSize,
                        double *V, int VSize,
                          double *M, int MSize,
                          double *Md, int MdSize,
                          double *dxa, int dxaSize,
                          double *dya, int dyaSize,
                          double *dza, int dzaSize,
                          int *index, int indexSize, 
                          int P, double kappa, int Nm, int LorY);
extern void multipole_sort( double *K, int KSize,
                            double *V, int VSize,
                            int *offTar, int offTarSize,
                            int *sizeTar, int sizeTarSize,
                            int *offMlt, int offMltSize,
                            double *M, int MSize,
                            double *Md, int MdSize,
                            double *xi, int xiSize,
                            double *yi, int yiSize,
                            double *zi, int ziSize,
                            double *xc, int xcSize,
                            double *yc, int ycSize,
                            double *zc, int zcSize,
                            int *index, int indexSize, 
                            int P, double kappa, int Nm, int LorY);
extern void multipoleKt_sort(double *Ktx, int KtxSize, 
                    double *Kty, int KtySize,
                    double *Ktz, int KtzSize,
                    int *offTar, int offTarSize,
                    int *sizeTar, int sizeTarSize,
                    int *offMlt, int offMltSize,
                    double *M , int MSize, 
                    double *xi, int xiSize, 
                    double *yi, int yiSize, 
                    double *zi, int ziSize,
                    double *xc, int xcSize, 
                    double *yc, int ycSize, 
                    double *zc, int zcSize,
                    int *index, int indexSize,
                    int P, double kappa, int Nm, int LorY);
extern void getIndex_arr(int P, int N,  
                        int *indices, int indicesSize,
                        int * ii    , int iiSize,
                        int * jj    , int jjSize,
                        int * kk    , int kkSize);
extern int setIndex(int P, int i, int j, int k);
extern void compute_potential_multipole(double *V, int VSize,
                                 double *xi , int xiSize, 
                                 double *yi , int yiSize, 
                                 double *zi , int ziSize,
                                 double *xc , int xcSize,
                                 double *yc , int ycSize, 
                                 double *zc , int zcSize,
                                 double *q  , int qSize,
                                 double *px , int pxSize,
                                 double *py , int pySize,
                                 double *pz , int pzSize,
                                 double *Qxx, int QxxSize,
                                 double *Qxy, int QxySize,
                                 double *Qxz, int QxzSize,
                                 double *Qyx, int QyxSize,
                                 double *Qyy, int QyySize,
                                 double *Qyz, int QyzSize,
                                 double *Qzx, int QzxSize,
                                 double *Qzy, int QzySize,
                                 double *Qzz, int QzzSize);
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1){(double *K, int KSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *Ktx, int KtxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *Kty, int KtySize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *Ktz, int KtzSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *V, int VSize)};
%apply (int* INPLACE_ARRAY1, int DIM1){(int *indices, int indicesSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *offTar, int offTarSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *sizeTar, int sizeTarSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *offMlt, int offMltSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *ii, int iiSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *jj, int jjSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *kk, int kkSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *index, int indexSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *M, int MSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Md, int MdSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *dxa, int dxaSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *dya, int dyaSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *dza, int dzaSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xi, int xiSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *yi, int yiSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *zi, int ziSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xc, int xcSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *yc, int ycSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *zc, int zcSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *q, int qSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *px, int pxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *py, int pySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *pz, int pzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qxx, int QxxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qxy, int QxySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qxz, int QxzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qyx, int QyxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qyy, int QyySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qyz, int QyzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qzx, int QzxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qzy, int QzySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qzz, int QzzSize)};
extern void multipole_c(double *K, int KSize,
                        double *V, int VSize,
                          double *M, int MSize,
                          double *Md, int MdSize,
                          double *dxa, int dxaSize,
                          double *dya, int dyaSize,
                          double *dza, int dzaSize,
                          int *index, int indexSize, 
                          int P, double kappa, int Nm, int LorY);
extern void multipole_sort( double *K, int KSize,
                            double *V, int VSize,
                            int *offTar, int offTarSize,
                            int *sizeTar, int sizeTarSize,
                            int *offMlt, int offMltSize,
                            double *M, int MSize,
                            double *Md, int MdSize,
                            double *xi, int xiSize,
                            double *yi, int yiSize,
                            double *zi, int ziSize,
                            double *xc, int xcSize,
                            double *yc, int ycSize,
                            double *zc, int zcSize,
                            int *index, int indexSize, 
                            int P, double kappa, int Nm, int LorY);
extern void multipoleKt_sort(double *Ktx, int KtxSize, 
                    double *Kty, int KtySize,
                    double *Ktz, int KtzSize,
                    int *offTar, int offTarSize,
                    int *sizeTar, int sizeTarSize,
                    int *offMlt, int offMltSize,
                    double *M , int MSize, 
                    double *xi, int xiSize, 
                    double *yi, int yiSize, 
                    double *zi, int ziSize,
                    double *xc, int xcSize, 
                    double *yc, int ycSize, 
                    double *zc, int zcSize,
                    int *index, int indexSize,
                    int P, double kappa, int Nm, int LorY);
extern void getIndex_arr(int P, int N, 
                        int *indices, int indicesSize,
                        int * ii    , int iiSize,
                        int * jj    , int jjSize,
                        int * kk    , int kkSize);
extern int setIndex(int P, int i, int j, int k);
extern void compute_potential_multipole(double *V, int VSize,
                                 double *xi , int xiSize, 
                                 double *yi , int yiSize, 
                                 double *zi , int ziSize,
                                 double *xc , int xcSize,
                                 double *yc , int ycSize, 
                                 double *zc , int zcSize,
                                 double *q  , int qSize,
                                 double *px , int pxSize,
                                 double *py , int pySize,
                                 double *pz , int pzSize,
                                 double *Qxx, int QxxSize,
                                 double *Qxy, int QxySize,
                                 double *Qxz, int QxzSize,
                                 double *Qyx, int QyxSize,
                                 double *Qyy, int QyySize,
                                 double *Qyz, int QyzSize,
                                 double *Qzx, int QzxSize,
                                 double *Qzy, int QzySize,
                                 double *Qzz, int QzzSize);
%clear (double *K, int KSize);
%clear (double *Ktx, int KtxSize);
%clear (double *Kty, int KtySize);
%clear (double *Ktz, int KtzSize);
%clear (double *V, int VSize);
%clear (double *M, int MSize);
%clear (double *Md, int MdSize);
%clear (double *dxa, int dxaSize);
%clear (double *dya, int dyaSize);
%clear (double *dza, int dzaSize);
%clear (double *xi, int xiSize);
%clear (double *yi, int yiSize);
%clear (double *zi, int ziSize);
%clear (double *xc, int xcSize);
%clear (double *yc, int ycSize);
%clear (double *zc, int zcSize);
%clear (double *q, int qSize);
%clear (double *px, int pxSize);
%clear (double *py, int pySize);
%clear (double *pz, int pzSize);
%clear (double *Qxx, int QxxSize);
%clear (double *Qxy, int QxySize);
%clear (double *Qxz, int QxzSize);
%clear (double *Qyx, int QyxSize);
%clear (double *Qyy, int QyySize);
%clear (double *Qyz, int QyzSize);
%clear (double *Qzx, int QzxSize);
%clear (double *Qzy, int QzySize);
%clear (double *Qzz, int QzzSize);
%clear (double *indices, int indicesSize);
%clear (int *offTar, int offTarSize);
%clear (int *sizeTar, int sizeTarSize);
%clear (int *offMlt, int offMltSize);
%clear (int *ii, int iiSize);
%clear (int *jj, int jjSize);
%clear (int *kk, int kkSize);
%clear (int *index, int indexSize);
