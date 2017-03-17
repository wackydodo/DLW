---
layout: page
title: "My New Page1"
category: doc
date: 2017-03-17 00:55:30
---


```python
import numpy as np
from tree import TreeModel

class BusinessAsUsual(object):
    def __init__(self, ghg=1000.0, emit_time=[0, 30, 60], emit_level=[52.0, 70.0, 81.4]):
        """
        emissions growth is assumed to slow down exogenously -- these assumptions
        represent an attempt to model emissions growth in a business-as-usual scenario
        that is in the absence of incentives
        """
        self.ghg = ghg
        self.emit_time = emit_time
        self.emit_level = emit_level
        self.emission_by_decisions = None
        self.emission_per_period = None
        self.emissions_to_ghg = None

    def emission_by_time(self, time):
        """
        return the bau emissions at any time t
        """
        if time < self.emit_time[1]:
            emissions = self.emit_level[0] + float(time) / (self.emit_time[1] - self.emit_time[0]) \
                        * (self.emit_level[1] - self.emit_level[0])
        elif time < self.emit_time[2]:
            emissions = self.emit_level[1] + float(time - self.emit_time[1]) / (self.emit_time[2] 
                        - self.emit_time[1]) * (self.emit_level[2] - self.emit_level[1])
        else:
            emissions = self.emit_level[2]
        return emissions

    def bau_emissions_setup(self, tree):
        """
        create default business as usual emissions path
        the emissions rate in each period are assumed to be the average of the emissions at the beginning
        and at the end of the period
        """
        num_periods = tree.num_periods
        self.emission_by_decisions = np.zeros(num_periods)
        self.emissions_per_period = np.zeros(num_periods)
        self.emissions_to_ghg = np.zeros(num_periods)
        self.emission_by_decisions[0] = self.emission_by_time(tree.decision_times[0])
        period_len = tree.decision_times[1:] - tree.decision_times[:-1]

        for n in range(1, num_periods):
            self.emission_by_decisions[n] = self.emission_by_time(tree.decision_times[n])
            self.emissions_per_period[n] = period_len[n] * (self.emission_by_decisions[n-1:n].mean())

        #the total increase in ghg level of 600 (from 400 to 1000) in the bau path is allocated over time
        self.emissions_to_ghg = 600 * self.emissions_per_period / self.emissions_per_period.sum()
```
