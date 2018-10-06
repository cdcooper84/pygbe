%module direct

%{
#define SWIG_FILE_WITH_INIT

extern void computeDiagonal(double *VL, int VLSize, double *KL, int KLSize, double *VY, int VYSize, double *KY, int KYSize, 
                    double *triangle, int triangleSize, double *centers, int centersSize, double kappa,
                    double K_diag, double V_diag, double *xk, int xkSize, double *wk, int wkSize); 

extern void direct_sort(double *K_aux, int K_auxSize, double *V_aux, int V_auxSize, int LorY, double K_diag, double V_diag, int IorE, double *triangle, int triangleSize, 
        int *tri, int triSize, int *k, int kSize, double *xi, int xiSize, double *yi, int yiSize, 
        double *zi, int ziSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, 
        double *s_zj, int s_zjSize,double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mKc, int mKcSize, double *mVc, int mVcSize,
        int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize, int *offSrc, int offSrcSize, int *offTwg, int offTwgSize,
        int *targets, int targetsSize, double *Area, int AreaSize, double *sglInt_int, int sglInt_intSize, double *sglInt_ext, int sglInt_extSize,
        double *xk, int xkSize, double *wk, int wkSize, double *Xsk, int XskSize, double *Wsk, int WskSize,
        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);

extern void directKt_sort(double *Ktx_aux, int Ktx_auxSize, double *Kty_aux, int Kty_auxSize, double *Ktz_aux, int Ktz_auxSize,
        int LorY, double *triangle, int triangleSize,
        int *k, int kSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, double *s_zj, int s_zjSize,
        double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mKc, int mKcSize,
        int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize,
        int *offSrc, int offSrcSize, int *offTwg, int offTwgSize, double *Area, int AreaSize,
        double *Xsk, int XskSize, double *Wsk, int WskSize, double kappa, double threshold, double eps, double *aux, int auxSize);

extern void direct_c(double *K_aux, int K_auxSize, double *V_aux, int V_auxSize, int LorY, double K_diag, double V_diag, int IorE, double *triangle, int triangleSize, 
        int *tri, int triSize, int *k, int kSize, double *xi, int xiSize, double *yi, int yiSize, 
        double *zi, int ziSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, 
        double *s_zj, int s_zjSize,double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mKc, int mKcSize, double *mVc, int mVcSize,
        int *targets, int targetsSize, double *Area, int AreaSize, double *sglInt_int, int sglInt_intSize, double *sglInt_ext, int sglInt_extSize,
        double *xk, int xkSize, double *wk, int wkSize, double *Xsk, int XskSize, double *Wsk, int WskSize,
        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);

extern void direct_c_derivative(double *dKx_aux, int dKx_auxSize, double *dKy_aux, int dKy_auxSize, double *dKz_aux, int dKz_auxSize, double *dVx_aux, int dVx_auxSize, double *dVy_aux, int dVy_auxSize, double *dVz_aux, int dVz_auxSize, int LorY, double K_diag, double V_diag, int IorE, double *triangle, int triangleSize, 
        int *tri, int triSize, int *k, int kSize, double *xi, int xiSize, double *yi, int yiSize, 
        double *zi, int ziSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, 
        double *s_zj, int s_zjSize,double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mKc, int mKcSize, double *mVc, int mVcSize,
        int *targets, int targetsSize, double *Area, int AreaSize, double *sglInt_int, int sglInt_intSize, double *sglInt_ext, int sglInt_extSize,
        double *xk, int xkSize, double *wk, int wkSize, double *Xsk, int XskSize, double *Wsk, int WskSize,
        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);

extern void coulomb_direct(double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize, 
                            double *m, int mSize, double *K_aux, int K_auxSize);

extern void direct_c_2derivative(double *dKxx_aux, int dKxx_auxSize, double *dKxy_aux, int dKxy_auxSize, double *dKxz_aux, int dKxz_auxSize,
                          double *dKyx_aux, int dKyx_auxSize, double *dKyy_aux, int dKyy_auxSize, double *dKyz_aux, int dKyz_auxSize,
                          double *dKzx_aux, int dKzx_auxSize, double *dKzy_aux, int dKzy_auxSize, double *dKzz_aux, int dKzz_auxSize,
                          double *dVxx_aux, int dVxx_auxSize, double *dVxy_aux, int dVxy_auxSize, double *dVxz_aux, int dVxz_auxSize,
                          double *dVyx_aux, int dVyx_auxSize, double *dVyy_aux, int dVyy_auxSize, double *dVyz_aux, int dVyz_auxSize,
                          double *dVzx_aux, int dVzx_auxSize, double *dVzy_aux, int dVzy_auxSize, double *dVzz_aux, int dVzz_auxSize,
                        int LorY, double K_diag, double V_diag, int IorE, double *triangle, int triangleSize,
                        int *tri, int triSize, int *k, int kSize, double *xi, int xiSize, double *yi, int yiSize, 
                        double *zi, int ziSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, 
                        double *s_zj, int s_zjSize, double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
                        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mKc, int mKcSize, double *mVc, int mVcSize,
                        int *targets, int targetsSize,double *Area, int AreaSize, double *sglInt_int, int sglInt_intSize, double *sglInt_ext, int sglInt_extSize, 
                        double *xk, int xkSize, double *wk, int wkSize, double *Xsk, int XskSize, double *Wsk, int WskSize, 
                        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);

extern void coulomb_energy_multipole(double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize, 
                        double *q, int qSize, double *px, int pxSize, double *py, int pySize, double *pz, int pzSize, 
                        double *px_pol, int px_polSize, double *py_pol, int py_polSize, double *pz_pol, int pz_polSize, 
                        double *Qxx, int QxxSize, double *Qxy, int QxySize, double *Qxz, int QxzSize, 
                        double *Qyx, int QyxSize, double *Qyy, int QyySize, double *Qyz, int QyzSize, 
                        double *Qzx, int QzxSize, double *Qzy, int QzySize, double *Qzz, int QzzSize, 
                        double *alphaxx, int alphaxxSize, double *thole, int tholeSize, 
                        int *polar_group, int polar_groupSize, 
                        int *connections_12, int connections_12Size, 
                        int *pointer_connections_12, int pointer_connections_12Size,
                        int *connections_13, int connections_13Size, 
                        int *pointer_connections_13, int pointer_connections_13Size,
                        double p12scale, double p13scale, double *K_aux, int K_auxSize);

extern void compute_induced_dipole(double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize, 
                        double *q, int qSize, double *px, int pxSize, double *py, int pySize, double *pz, int pzSize, 
                        double *px_pol, int px_polSize, double *py_pol, int py_polSize, double *pz_pol, int pz_polSize, 
                        double *Qxx, int QxxSize, double *Qxy, int QxySize, double *Qxz, int QxzSize, 
                        double *Qyx, int QyxSize, double *Qyy, int QyySize, double *Qyz, int QyzSize, 
                        double *Qzx, int QzxSize, double *Qzy, int QzySize, double *Qzz, int QzzSize, 
                        double *alphaxx, int alphaxxSize, double *alphaxy, int alphaxySize, double *alphaxz, int alphaxzSize, 
                        double *alphayx, int alphayxSize, double *alphayy, int alphayySize, double *alphayz, int alphayzSize, 
                        double *alphazx, int alphazxSize, double *alphazy, int alphazySize, double *alphazz, int alphazzSize, 
                        double *thole, int tholeSize, int *polar_group, int polar_groupSize, 
                        int *connections_12, int connections_12Size, 
                        int *pointer_connections_12, int pointer_connections_12Size,
                        int *connections_13, int connections_13Size, 
                        int *pointer_connections_13, int pointer_connections_13Size,
                        double *dphix_reac, int dphix_reacSize, double *dphiy_reac, int dphiy_reacSize, double *dphiz_reac, int dphiz_reacSize, double E);

%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1){(double *K_aux, int K_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKx_aux, int dKx_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKy_aux, int dKy_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKz_aux, int dKz_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKxx_aux, int dKxx_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKxy_aux, int dKxy_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKxz_aux, int dKxz_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKyx_aux, int dKyx_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKyy_aux, int dKyy_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKyz_aux, int dKyz_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKzx_aux, int dKzx_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKzy_aux, int dKzy_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dKzz_aux, int dKzz_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *Ktx_aux, int Ktx_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *Kty_aux, int Kty_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *Ktz_aux, int Ktz_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *V_aux, int V_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVx_aux, int dVx_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVy_aux, int dVy_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVz_aux, int dVz_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVxx_aux, int dVxx_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVxy_aux, int dVxy_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVxz_aux, int dVxz_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVyx_aux, int dVyx_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVyy_aux, int dVyy_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVyz_aux, int dVyz_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVzx_aux, int dVzx_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVzy_aux, int dVzy_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dVzz_aux, int dVzz_auxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *KL, int KLSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *VL, int VLSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *KY, int KYSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *VY, int VYSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *triangle, int triangleSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *centers, int centersSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *tri, int triSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *k, int kSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *interList, int interListSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *offTar, int offTarSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *sizeTar, int sizeTarSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *offSrc, int offSrcSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *offTwg, int offTwgSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *s_xj, int s_xjSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *s_yj, int s_yjSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *s_zj, int s_zjSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xi, int xiSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *yi, int yiSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *zi, int ziSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xt, int xtSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *yt, int ytSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *zt, int ztSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *m, int mSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *mx, int mxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *my, int mySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *mz, int mzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *q, int qSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *px, int pxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *py, int pySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *pz, int pzSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *px_pol, int px_polSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *py_pol, int py_polSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *pz_pol, int pz_polSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *dphix_reac, int dphix_reacSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *dphiy_reac, int dphiy_reacSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *dphiz_reac, int dphiz_reacSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qxx, int QxxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qxy, int QxySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qxz, int QxzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qyx, int QyxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qyy, int QyySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qyz, int QyzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qzx, int QzxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qzy, int QzySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Qzz, int QzzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *alphaxx, int alphaxxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *alphaxy, int alphaxySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *alphaxz, int alphaxzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *alphayx, int alphayxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *alphayy, int alphayySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *alphayz, int alphayzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *alphazx, int alphazxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *alphazy, int alphazySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *alphazz, int alphazzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *thole, int tholeSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *polar_group, int polar_groupSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *connections_12, int connections_12Size)};
%apply (int* IN_ARRAY1, int DIM1){(int *pointer_connections_12, int pointer_connections_12Size)};
%apply (int* IN_ARRAY1, int DIM1){(int *connections_13, int connections_13Size)};
%apply (int* IN_ARRAY1, int DIM1){(int *pointer_connections_13, int pointer_connections_13Size)};
%apply (double* IN_ARRAY1, int DIM1){(double *mKc, int mKcSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *mVc, int mVcSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *targets, int targetsSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Area, int AreaSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *sglInt_int, int sglInt_intSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *sglInt_ext, int sglInt_extSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xk, int xkSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *wk, int wkSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Xsk, int XskSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Wsk, int WskSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *aux, int auxSize)};

extern void computeDiagonal(double *VL, int VLSize, double *KL, int KLSize, double *VY, int VYSize, double *KY, int KYSize, 
                    double *triangle, int triangleSize, double *centers, int centersSize, double kappa,
                    double K_diag, double V_diag, double *xk, int xkSize, double *wk, int wkSize); 

extern void direct_sort(double *K_aux, int K_auxSize, double *V_aux, int V_auxSize, int LorY, double K_diag, double V_diag, int IorE, double *triangle, int triangleSize, 
        int *tri, int triSize, int *k, int kSize, double *xi, int xiSize, double *yi, int yiSize, 
        double *zi, int ziSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, 
        double *s_zj, int s_zjSize,double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mKc, int mKcSize, double *mVc, int mVcSize,
        int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize, int *offSrc, int offSrcSize, int *offTwg, int offTwgSize,
        int *targets, int targetsSize, double *Area, int AreaSize, double *sglInt_int, int sglInt_intSize, double *sglInt_ext, int sglInt_extSize, 
        double *xk, int xkSize, double *wk, int wkSize, double *Xsk, int XskSize, double *Wsk, int WskSize,
        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);

extern void directKt_sort(double *Ktx_aux, int Ktx_auxSize, double *Kty_aux, int Kty_auxSize, double *Ktz_aux, int Ktz_auxSize,
        int LorY, double *triangle, int triangleSize,
        int *k, int kSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, double *s_zj, int s_zjSize,
        double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mKc, int mKcSize,
        int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize,
        int *offSrc, int offSrcSize, int *offTwg, int offTwgSize, double *Area, int AreaSize,
        double *Xsk, int XskSize, double *Wsk, int WskSize, double kappa, double threshold, double eps, double *aux, int auxSize);

extern void direct_c(double *K_aux, int K_auxSize, double *V_aux, int V_auxSize, int LorY, double K_diag, double V_diag, int IorE, double *triangle, int triangleSize, 
        int *tri, int triSize, int *k, int kSize, double *xi, int xiSize, double *yi, int yiSize, 
        double *zi, int ziSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, 
        double *s_zj, int s_zjSize,double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mKc, int mKcSize, double *mVc, int mVcSize,
        int *targets, int targetsSize, double *Area, int AreaSize, double *sglInt_int, int sglInt_intSize, double *sglInt_ext, int sglInt_extSize,
        double *xk, int xkSize, double *wk, int wkSize, double *Xsk, int XskSize, double *Wsk, int WskSize, 
        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);

extern void direct_c_derivative(double *dKx_aux, int dKx_auxSize, double *dKy_aux, int dKy_auxSize, double *dKz_aux, int dKz_auxSize, double *dVx_aux, int dVx_auxSize, double *dVy_aux, int dVy_auxSize, double *dVz_aux, int dVz_auxSize, int LorY, double K_diag, double V_diag, int IorE, double *triangle, int triangleSize, 
        int *tri, int triSize, int *k, int kSize, double *xi, int xiSize, double *yi, int yiSize, 
        double *zi, int ziSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, 
        double *s_zj, int s_zjSize,double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mKc, int mKcSize, double *mVc, int mVcSize,
        int *targets, int targetsSize, double *Area, int AreaSize, double *sglInt_int, int sglInt_intSize, double *sglInt_ext, int sglInt_extSize,
        double *xk, int xkSize, double *wk, int wkSize, double *Xsk, int XskSize, double *Wsk, int WskSize, 
        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);

extern void direct_c_2derivative(double *dKxx_aux, int dKxx_auxSize, double *dKxy_aux, int dKxy_auxSize, double *dKxz_aux, int dKxz_auxSize,
                          double *dKyx_aux, int dKyx_auxSize, double *dKyy_aux, int dKyy_auxSize, double *dKyz_aux, int dKyz_auxSize,
                          double *dKzx_aux, int dKzx_auxSize, double *dKzy_aux, int dKzy_auxSize, double *dKzz_aux, int dKzz_auxSize,
                          double *dVxx_aux, int dVxx_auxSize, double *dVxy_aux, int dVxy_auxSize, double *dVxz_aux, int dVxz_auxSize,
                          double *dVyx_aux, int dVyx_auxSize, double *dVyy_aux, int dVyy_auxSize, double *dVyz_aux, int dVyz_auxSize,
                          double *dVzx_aux, int dVzx_auxSize, double *dVzy_aux, int dVzy_auxSize, double *dVzz_aux, int dVzz_auxSize,
                        int LorY, double K_diag, double V_diag, int IorE, double *triangle, int triangleSize,
                        int *tri, int triSize, int *k, int kSize, double *xi, int xiSize, double *yi, int yiSize, 
                        double *zi, int ziSize, double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, 
                        double *s_zj, int s_zjSize, double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
                        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mKc, int mKcSize, double *mVc, int mVcSize,
                        int *targets, int targetsSize,double *Area, int AreaSize, double *sglInt_int, int sglInt_intSize, double *sglInt_ext, int sglInt_extSize, 
                        double *xk, int xkSize, double *wk, int wkSize, double *Xsk, int XskSize, double *Wsk, int WskSize, 
                        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);
extern void coulomb_direct(double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize, 
                            double *m, int mSize, double *K_aux, int K_auxSize);

extern void coulomb_energy_multipole(double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize, 
                        double *q, int qSize, double *px, int pxSize, double *py, int pySize, double *pz, int pzSize, 
                        double *px_pol, int px_polSize, double *py_pol, int py_polSize, double *pz_pol, int pz_polSize, 
                        double *Qxx, int QxxSize, double *Qxy, int QxySize, double *Qxz, int QxzSize, 
                        double *Qyx, int QyxSize, double *Qyy, int QyySize, double *Qyz, int QyzSize, 
                        double *Qzx, int QzxSize, double *Qzy, int QzySize, double *Qzz, int QzzSize, 
                        double *alphaxx, int alphaxxSize, double *thole, int tholeSize, 
                        int *polar_group, int polar_groupSize, 
                        int *connections_12, int connections_12Size, 
                        int *pointer_connections_12, int pointer_connections_12Size,
                        int *connections_13, int connections_13Size, 
                        int *pointer_connections_13, int pointer_connections_13Size,
                        double p12scale, double p13scale, double *K_aux, int K_auxSize);

extern void compute_induced_dipole(double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize, 
                        double *q, int qSize, double *px, int pxSize, double *py, int pySize, double *pz, int pzSize, 
                        double *px_pol, int px_polSize, double *py_pol, int py_polSize, double *pz_pol, int pz_polSize, 
                        double *Qxx, int QxxSize, double *Qxy, int QxySize, double *Qxz, int QxzSize, 
                        double *Qyx, int QyxSize, double *Qyy, int QyySize, double *Qyz, int QyzSize, 
                        double *Qzx, int QzxSize, double *Qzy, int QzySize, double *Qzz, int QzzSize, 
                        double *alphaxx, int alphaxxSize, double *alphaxy, int alphaxySize, double *alphaxz, int alphaxzSize, 
                        double *alphayx, int alphayxSize, double *alphayy, int alphayySize, double *alphayz, int alphayzSize, 
                        double *alphazx, int alphazxSize, double *alphazy, int alphazySize, double *alphazz, int alphazzSize, 
                        double *thole, int tholeSize, int *polar_group, int polar_groupSize, 
                        int *connections_12, int connections_12Size, 
                        int *pointer_connections_12, int pointer_connections_12Size,
                        int *connections_13, int connections_13Size, 
                        int *pointer_connections_13, int pointer_connections_13Size,
                        double *dphix_reac, int dphix_reacSize, double *dphiy_reac, int dphiy_reacSize, double *dphiz_reac, int dphiz_reacSize, double E);

%clear (double *K_aux, int K_auxSize); 
%clear (double *dKx_aux, int dKx_auxSize); 
%clear (double *dKy_aux, int dKy_auxSize); 
%clear (double *dKz_aux, int dKz_auxSize); 
%clear (double *dKxx_aux, int dKxx_auxSize); 
%clear (double *dKxy_aux, int dKxy_auxSize); 
%clear (double *dKxz_aux, int dKxz_auxSize); 
%clear (double *dKyx_aux, int dKyx_auxSize); 
%clear (double *dKyy_aux, int dKyy_auxSize); 
%clear (double *dKyz_aux, int dKyz_auxSize); 
%clear (double *dKzx_aux, int dKzx_auxSize); 
%clear (double *dKzy_aux, int dKzy_auxSize); 
%clear (double *dKzz_aux, int dKzz_auxSize); 
%clear (double *Ktx_aux, int Ktx_auxSize); 
%clear (double *Kty_aux, int Kty_auxSize); 
%clear (double *Ktz_aux, int Ktz_auxSize); 
%clear (double *V_aux, int V_auxSize); 
%clear (double *dVx_aux, int dVx_auxSize); 
%clear (double *dVy_aux, int dVy_auxSize); 
%clear (double *dVz_aux, int dVz_auxSize); 
%clear (double *dVxx_aux, int dVxx_auxSize); 
%clear (double *dVxy_aux, int dVxy_auxSize); 
%clear (double *dVxz_aux, int dVxz_auxSize); 
%clear (double *dVyx_aux, int dVyx_auxSize); 
%clear (double *dVyy_aux, int dVyy_auxSize); 
%clear (double *dVyz_aux, int dVyz_auxSize); 
%clear (double *dVzx_aux, int dVzx_auxSize); 
%clear (double *dVzy_aux, int dVzy_auxSize); 
%clear (double *dVzz_aux, int dVzz_auxSize); 
%clear (double *KL, int KLSize); 
%clear (double *VL, int VLSize); 
%clear (double *KY, int KYSize); 
%clear (double *VY, int VYSize); 
%clear (double *triangle, int triangleSize); 
%clear (double *centers, int centersSize); 
%clear (int *tri, int triSize); 
%clear (int *k, int kSize); 
%clear (int *interList, int interListSize); 
%clear (int *offTar, int offTarSize); 
%clear (int *sizeTar, int sizeTarSize); 
%clear (int *offSrc, int offSrcSize); 
%clear (int *offTwg, int offTwgSize); 
%clear (double *s_xj, int s_xjSize); 
%clear (double *s_yj, int s_yjSize); 
%clear (double *s_zj, int s_zjSize); 
%clear (double *xt, int xtSize); 
%clear (double *yt, int ytSize); 
%clear (double *zt, int ztSize); 
%clear (double *xi, int xiSize); 
%clear (double *yi, int yiSize); 
%clear (double *zi, int ziSize); 
%clear (double *m, int mSize); 
%clear (double *mx, int mxSize); 
%clear (double *my, int mySize); 
%clear (double *mz, int mzSize);
%clear (double *q, int qSize); 
%clear (double *px, int pxSize); 
%clear (double *py, int pySize); 
%clear (double *pz, int pzSize);
%clear (double *px_pol, int px_polSize); 
%clear (double *py_pol, int py_polSize); 
%clear (double *pz_pol, int pz_polSize);
%clear (double *dphix_reac, int dphix_reacSize); 
%clear (double *dphiy_reac, int dphiy_reacSize); 
%clear (double *dphiz_reac, int dphiz_reacSize);
%clear (double *Qxx, int QxxSize); 
%clear (double *Qxy, int QxySize); 
%clear (double *Qxz, int QxzSize);
%clear (double *Qyx, int QyxSize); 
%clear (double *Qyy, int QyySize); 
%clear (double *Qyz, int QyzSize);
%clear (double *Qzx, int QzxSize); 
%clear (double *Qzy, int QzySize); 
%clear (double *Qzz, int QzzSize);
%clear (double *alphaxx, int alphaxxSize); 
%clear (double *alphaxy, int alphaxySize); 
%clear (double *alphaxz, int alphaxzSize);
%clear (double *alphayx, int alphayxSize); 
%clear (double *alphayy, int alphayySize); 
%clear (double *alphayz, int alphayzSize);
%clear (double *alphazx, int alphazxSize); 
%clear (double *alphazy, int alphazySize); 
%clear (double *alphazz, int alphazzSize);
%clear (double *thole, int tholeSize);
%clear (int *polar_group, int polar_groupSize);
%clear (int *connections_12, int connections_12Size);
%clear (int *pointer_connections_12, int pointer_connections_12Size);
%clear (int *connections_13, int connections_13Size);
%clear (int *pointer_connections_13, int pointer_connections_13Size);
%clear (double *mKc, int mKcSize);
%clear (double *mVc, int mVcSize);
%clear (int *targets, int targetsSize); 
%clear (double *Area, int AreaSize); 
%clear (double *sglInt_int, int sglInt_intSize); 
%clear (double *sglInt_ext, int sglInt_extSize); 
%clear (double *xk, int xkSize); 
%clear (double *wk, int wkSize); 
%clear (double *Xsk, int XskSize); 
%clear (double *Wsk, int WskSize); 
%clear (double *aux, int auxSize); 
