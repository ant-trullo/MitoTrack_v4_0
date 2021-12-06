"""This function trackes nuclei for very big data.

It just coordinates the previous function 'NucleiConnect' to work in a
multiprocessing pool.
"""


import multiprocessing
import numpy as np

import NucleiConnect


class NucleiConnectMultiCore:
    """The only class, does all the job"""
    def __init__(self, nuclei_seg, dist_thr):

        steps   =  nuclei_seg.shape[0]
        cpu_ow  =  multiprocessing.cpu_count()
        chops   =  np.round(np.linspace(0, steps, cpu_ow + 1)).astype(np.int)

        if steps > 10 * cpu_ow:
            job_args  =  []
            for t in range(chops.size - 1):
                job_args.append([nuclei_seg[chops[t]:chops[t + 1], :, :], dist_thr])

            pool     =  multiprocessing.Pool()
            results  =  pool.map(NucleiConnect.NucleiConnect, job_args)
            pool.close()

            nuclei_tracked                           =  np.zeros(nuclei_seg.shape, dtype=np.int16)
            nuclei_tracked[chops[0]:chops[1], :, :]  =  results[0].nuclei_tracked

            for t in range(1, cpu_ow):
                idx  =  np.unique(results[t].nuclei_tracked[0, :, :])[1:]                                       # after pooling, results must be concatenate but saving the correct tag for each nucleus:
                for k in idx:                                                                                   # here we work at the interface (last frame versus first frame of the following results block)
                    tag  =  (nuclei_tracked[chops[t] - 1, :, :] * (results[t].nuclei_tracked[0, :, :] == k))
                    tag  =  tag.reshape(tag.size)
                    tag  =  np.delete(tag, np.where(tag == 0))
                    if tag.size > 0:
                        tag  =  np.median(tag)
                        nuclei_tracked[chops[t]:chops[t + 1], :, :]  +=  ((results[t].nuclei_tracked == k) * tag).astype(np.int)

        else:
            nuclei_tracked  =  NucleiConnect.NucleiConnect([nuclei_seg, dist_thr]).nuclei_tracked


        self.nuclei_tracked  =  nuclei_tracked
