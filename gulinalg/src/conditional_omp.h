/* Header file to conditionally wrap omp.h defines
 *
 * _OPENMP should be defined if omp.h is safe to include
 */
#if defined(_OPENMP)
#include <omp.h>
#define have_openmp 1
#else
/* These are fake defines to make these symbols valid in the c code
 *
 * Uses of these symbols should be within ``if (have_openmp)`` blocks:
 *
 *     if (have_openmp){
 *         omp_set_num_threads(nthreads)
 *     }
 * */
typedef int omp_lock_t;
void omp_init_lock(omp_lock_t *lock) {};
void omp_destroy_lock(omp_lock_t *lock) {};
void omp_set_lock(omp_lock_t *lock) {};
void omp_unset_lock(omp_lock_t *lock) {};
int omp_test_lock(omp_lock_t *lock) {return 1;};
void omp_set_dynamic(int dynamic_threads) {};
void omp_set_num_threads(int num_threads) {};
int omp_get_num_procs() {return 1;};
int omp_get_max_threads() {return 1;};
int omp_get_num_threads() {return 1;};
int omp_get_thread_num() {return 1;};
#define have_openmp 0
#endif
