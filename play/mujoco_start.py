# run in mujoco_py==1.5.0
import mujoco_py
import os
import numpy
from mujoco_py import load_model_from_path, MjSim, MjViewer

mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
print('xml_path',xml_path)
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = MjViewer(sim)

for i in range(10000):
    sim.step()
    viewer.render()
    if i%20==0:
        print(sim.data.xfrc_applied[1])
