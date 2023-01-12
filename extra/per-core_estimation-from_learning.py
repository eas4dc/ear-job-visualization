#!/usr/bin/env python
# coding: utf-8

# In[18]:


import matplotlib.pyplot as plt

import struct
# import pandas as pd
import sys
import os
import numpy as np

# Needed to import my common API
sys.path.insert(0, '../common')

import io_api
import utils


# ### The working tool

# In[16]:


path = '/home/xovidal/visualization/files'
file = 'learning_phase.csv'

df = (io_api
      .read_data(os.path.join(path, file))
      .pipe(utils.filter_df)
      .assign(
          IPC=lambda df: 1 / df.CPI,
          # VPI = lambda df:
          )
      )
# df.columns


# In[17]:


num_cpus = 48
cpus_node = 48

with open('../files/coeffs.cpumodel.6126', mode='rb') as coeffs:
    ipc_c, gbs_c, vpi_c, avg_freq_c, inter = struct.unpack_from('ddddd',
                                                                coeffs.read())
    print('Working with coeffs (IPC/GBS/VPI/AVG_CPUFREQ)'
          f' {ipc_c}/{gbs_c}/{vpi_c}/{avg_freq_c}'
          f' Intercept: {inter}')

    df_grouped = (df
                  .groupby(['JOBNAME', 'DEF_FREQ_KHZ'])
                  .mean()
                  .drop(index='dgemm_example', level=0)  # Dropped because of the lack of VPI info
                  .assign(
                      VPI=0,
                      avg_cpufreq=lambda df: df.AVG_CPUFREQ_KHZ / 1000000,
                      estimated_power=lambda df: ((df['IPC'] * ipc_c) +
                                                  (df['MEM_GBS'] * gbs_c) +
                                                  (df['VPI'] * vpi_c) +
                                                  (df['avg_cpufreq'] * avg_freq_c)
                                                  + inter) * num_cpus / cpus_node
                    )
                  )
    df_grouped.to_csv(os.path.join(path, 'estimated_power.csv'),
                      columns=['DC_NODE_POWER_W', 'estimated_power'])

    apps = np.unique([x for x, y in df_grouped.index])
    freqs = np.unique([y for x, y in df_grouped.index])

    for app_idx, app in enumerate(apps):

        fig, axes = plt.subplots(1, freqs.size, sharey=True, figsize=(19.2, 4.8))
        fig.suptitle(app)

        for freq_idx, freq in enumerate(freqs):
            curr_df = df_grouped.loc[app, freq]

            bars = axes[freq_idx].bar(['Estimated', 'Observed'], [curr_df['estimated_power'],
                                                                  curr_df['DC_NODE_POWER_W']])
            axes[freq_idx].set_title(freq)

        # plt.savefig(os.path.join(path, f'{app}.png'))
    # plt.show()


# In[ ]:





# In[ ]:




