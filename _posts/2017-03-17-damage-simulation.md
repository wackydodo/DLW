---
layout: page
title: "Damage Simulation"
category: cls
date: 2017-03-17 04:24:59
---


```{r eval=FALSE}
from tree import TreeModel

tree = TreeModel()
test = DamageSimulation(tree, ghg_levels=[450, 650, 1000], draws=400, peak_temp=6.0, disaster_tail=18.0,
       tip_on=True, temp_map=1, temp_dist_params=None, pindyck_impact_k=4.5, pindyck_impact_theta=21341.0,       pindyck_impact_displace=-0.0000746, maxh=100.0, cons_growth=0.015)
test.simulate()
```
<br>
