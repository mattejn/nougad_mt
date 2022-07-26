#include <R.h>
#include <R_ext/Rdynload.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct thread_bounds
{
  int start, stop;
  const int *dp_c, *kp_c, *iters_c;
  const float *lr_c, *acc_c, *s_dk_c, *spw_dk_c, *snw_dk_c, *nw_k_c, *y_dn_c;
  float *x_kn_c, *r_dn_c;
} ThreadData;
void *
cell_gd(void *td)
{
  ThreadData *data = (ThreadData *)td;
  const size_t d = *data->dp_c, k = *data->kp_c, iters = *data->iters_c;
  const float alpha = *data->lr_c, accel = *data->acc_c;
  size_t ni, di, ki, ii;
  float *lastg_k =
    malloc(sizeof(float) * k); // allocate vector of size k for grads
  for (ni = data->start; ni < data->stop; ++ni) { // for each cell (row)
    float *__restrict__ x_k =
      data->x_kn_c + ni * k; // declare x_k as the sole pointer to the
                             // abundandance vec(len=k) for curr cell
    float *__restrict__ r_d =
      data->r_dn_c + ni * d; // declare r_d as the sole pointer to the residual
                             // vec (len=d) for curr cell
    const float *__restrict__ y_d =
      data->y_dn_c + ni * d; // declare y_d as the sole pointer to the mixed
                             // vec(len=d) for curr cell
    for (ki = 0; ki < k; ++ki) // first set vals for all prev grads to 0
      lastg_k[ki] = 0;
    for (ii = 0;; ++ii) {        // for each iteration
      for (di = 0; di < d; ++di) // for each detector
        r_d[di] = -y_d[di];      // set the residual to negative of mixed value
      for (ki = 0; ki < k; ++ki) // for each spectrum
        for (di = 0; di < d; ++di) // for each detector in that spectrum
          r_d[di] +=
            x_k[ki] *
            data
              ->s_dk_c[di + d * ki]; // increment the residual in that detector
      // by the curr spectrum val in that detector * curr spectrum abundance

      if (ii >= iters)
        break; // end if we reached maxiters
      // realistically we will never reach 0 residuals with unmixing on real
      // data so that check is redundant

      for (ki = 0; ki < k; ++ki) { // for each spectra calculate gradient
        float gki =
          (x_k[ki] > 0
             ? 0
             : data->nw_k_c[ki] * x_k[ki]); // if abundance for curr spec is neg
                                            // add ab*nw(for that spec) to grad
        for (di = 0; di < d; ++di) // for each detector in that spectrum
          gki += // if residual in that detector is pos add res*spw to grad,
                 // otherwise res*snw
            r_d[di] * (r_d[di] > 0 ? data->spw_dk_c[di + d * ki]
                                   : data->snw_dk_c[di + d * ki]);

        gki *= alpha;              // multiply grad by learning rate
        if (gki * lastg_k[ki] > 0) // if the grad goes in the same dir as prev
          gki += accel * lastg_k[ki]; // add prev grad*accel to curr grad
        x_k[ki] -= gki; // step against the gradient (decrease curr spec
                        // abundance by grad)
        lastg_k[ki] = gki; // save grad curr as last
      }
    }
  }

  free(lastg_k); // free memory used by last grad vec
  return 0;
}

void
nougadmt_c(const int *np,       /* nrow of observation mtx -> n cells */
           const int *dp,       /* num of detectors */
           const int *kp,       /* number of fluorochromes (nrow spectra) */
           const int *itersp,   /* number of iterations */
           const int *nthreads, // number of threads
           const float *alphap, /* learning rate */
           const float *accelp, /*acceleration */
           const float *s_dk,   // spectra
           const float *spw_dk, // spectra*spw
           const float *snw_dk, // spectra*snw
           const float *nw_k,   // negative abundance weights for each spectrum
           const float *y_dn,   // mixed data
           float *x_kn,         // starting abundances (mtx)
           float *r_dn)         // starting residuals (mtx of 0s)
{

  const size_t threads = *nthreads, n = *np; // get thread number
  pthread_t thread[threads];
  struct thread_bounds bounds[threads];
  int cells_per_thread = ceil(n / threads); // get rounded cells per thread
  int i;                          // split the data to chunks for each thread
  for (i = 0; i < threads; i++) { // pass everything to struct for each thread
    bounds[i].start = i * cells_per_thread;
    bounds[i].stop = (i + 1) * cells_per_thread;
    bounds[i].acc_c = accelp;
    bounds[i].lr_c = alphap;
    bounds[i].dp_c = dp;
    bounds[i].iters_c = itersp;
    bounds[i].kp_c = kp;
    bounds[i].nw_k_c = nw_k;
    bounds[i].r_dn_c = r_dn;
    bounds[i].snw_dk_c = snw_dk;
    bounds[i].spw_dk_c = spw_dk;
    bounds[i].s_dk_c = s_dk;
    bounds[i].x_kn_c = x_kn;
    bounds[i].y_dn_c = y_dn;
  }
  bounds[threads - 1].stop =
    n; // make sure we don't overshoot actual data length
  for (i = 0; i < threads;
       i++) { // spawn threads and run their respective chunks
    pthread_create(&thread[i], NULL, *cell_gd, &bounds[i]);
  }
  for (i = 0; i < threads; i++) { // wait for all threads to properly exit
    pthread_join(thread[i], NULL);
  }
}
static const R_CMethodDef cMethods[] = {
  { "nougadmt_c", (DL_FUNC)&nougadmt_c, 14 },
  { NULL, NULL, 0 }
};

void
R_init_nougadmt(DllInfo *info)
{
  R_registerRoutines(info, cMethods, NULL, NULL, NULL);
  R_useDynamicSymbols(info, FALSE);
}
