#ifndef TQ_MEMBW_H
#define TQ_MEMBW_H

/*
 * membw.h - meet de maximale lees-bandbreedte van het RAM per threadaantal.
 *
 * membw_measure(threads, n, out_gbs) vult out_gbs[i] met GB/s voor threads[i].
 * Geeft 0 terug bij succes, -1 als er geen geheugen voor de buffer was
 * (out_gbs is dan op 0 gezet).
 *
 * Elke thread leest een eigen aaneengesloten stuk van een buffer die veel groter
 * is dan de L3-cache. De buffer wordt per thread "first touch" ingevuld, zodat
 * NUMA-systemen de pagina's netjes verdelen. Beste van MEMBW_REPS runs.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#define MEMBW_REPS 5

static volatile uint64_t g_membw_sink;

static inline uint64_t membw_read_range(const uint64_t *p, size_t n) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 += p[i];     a1 += p[i + 1]; a2 += p[i + 2]; a3 += p[i + 3];
        a4 += p[i + 4]; a5 += p[i + 5]; a6 += p[i + 6]; a7 += p[i + 7];
    }
    for (; i < n; i++) a0 += p[i];
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

static inline int membw_measure(const int *threads, int n_threads, double *out_gbs) {
    static const size_t sizes_mb[] = {512, 256, 128, 64};
    uint64_t *buf = NULL;
    size_t    n   = 0;

    for (int t = 0; t < n_threads; t++) out_gbs[t] = 0.0;

    for (size_t s = 0; s < sizeof sizes_mb / sizeof sizes_mb[0]; s++) {
        n   = sizes_mb[s] * 1024u * 1024u / sizeof(uint64_t);
        buf = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (buf) break;
    }
    if (!buf) return -1;

    int max_t = 1;
    for (int t = 0; t < n_threads; t++) if (threads[t] > max_t) max_t = threads[t];

    /* first touch, met niet-nul data */
    #pragma omp parallel num_threads(max_t)
    {
        const size_t nt    = (size_t)omp_get_num_threads();
        const size_t id    = (size_t)omp_get_thread_num();
        const size_t chunk = n / nt;
        const size_t start = id * chunk;
        const size_t end   = (id == nt - 1) ? n : start + chunk;
        for (size_t i = start; i < end; i++) buf[i] = (uint64_t)i * 0x9E3779B97F4A7C15ull + 1;
    }

    const double bytes = (double)n * (double)sizeof(uint64_t);

    for (int t = 0; t < n_threads; t++) {
        const int nthr = threads[t] >= 1 ? threads[t] : 1;
        double best = 0.0;

        for (int rep = 0; rep < MEMBW_REPS + 1; rep++) {
            uint64_t total = 0;
            const double t0 = omp_get_wtime();

            #pragma omp parallel num_threads(nthr) reduction(+:total)
            {
                const size_t nt    = (size_t)omp_get_num_threads();
                const size_t id    = (size_t)omp_get_thread_num();
                const size_t chunk = n / nt;
                const size_t start = id * chunk;
                const size_t end   = (id == nt - 1) ? n : start + chunk;
                total += membw_read_range(buf + start, end - start);
            }

            const double dt = omp_get_wtime() - t0;
            g_membw_sink = total;
            if (rep == 0 || dt <= 0.0) continue; /* eerste run = warmup */

            const double gbs = bytes / dt / 1e9;
            if (gbs > best) best = gbs;
        }
        out_gbs[t] = best;
    }

    free(buf);
    return 0;
}

#endif /* TQ_MEMBW_H */