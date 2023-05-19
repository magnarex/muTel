import os
import json
import numpy as np
import itertools
# token laptop: 2678d4463c3c471fbb051fea50561272
# token pc: 0ca788318dd5416595d71d496db61e20
# parent = '/afs/ciemat.es/user/m/martialc/public/muTel/'
# parent = 'J:/public/muTel/'
parent = os.path.abspath(__file__).replace("\\",'/').split('src/muTel')[0]

superlayers = set([1,2,3,4])
layers = set([1,2,3,4])

ns = 1e-9


vdrift = 55e-3 #FIXME: mm/ns?
cell_height = 13 # mm
cell_width = 2*21 # mm
ncells = 16

sl_gap = 193 ##mm
sheet_height = 1 #mm (aprox)

sl_height = -(2*sheet_height+4*cell_height)*np.array([0,1,0,1]) - sl_gap*np.array([0,0,1,1])
wire_height = -1*cell_height*np.arange(len(layers))-cell_height*0.5
# layer_offset = -cell_width*np.array((0,0.5,0,0.5))  # (l1,l2,l3,l4)
layer_offset = -cell_width*np.array((0.5,0,0.5,0))  # (l4,l3,l2,l1)

with open(f'{parent}/src/muTel/dqm/config/pattern.json','r') as file:
    patt_dict = json.load(file)

with open(f'{parent}/src/muTel/dqm/config/laterality.json','r') as file:
    lat_dict = json.load(file)

all_lats = list(map(lambda x: ''.join(x), itertools.product('LR',repeat=len(layers))))

