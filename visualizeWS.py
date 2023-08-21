import klampt
from klampt import WorldModel, vis, Geometry3D
import math,time
import klampt.model.workspace
from copy import deepcopy
from klampt.model.collide import WorldCollider
from klampt.model import ik
from klampt.math import vectorops as vo
world = WorldModel()
world.loadElement('../../../Models/robots/Diphtheria_test.urdf')
vis.add('world',world)
robot = world.robot(0)
collider = WorldCollider(world)
N_links = world.numRobotLinks(0)
print(N_links)
for i in range(N_links):
    print(robot.link(i).name, robot.link(i).index)


initial_config = robot.getConfig()
# print(initial_config[17],initial_config[31])
initial_config[18] = math.pi/2
#initial_config[32] = math.pi/2
robot.setConfig(initial_config)
cols = collider.robotSelfCollisions()
for c in cols:
    print('collision:',c[0].name,c[1].name)


(low,high) = robot.getJointLimits()
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    low[i] = 0.
    high[i] = 0.
robot.setJointLimits(low,high)
print(robot.getJointLimits())

print(robot.link(20).getWorldPosition([0,0,0]))
## Compute WS
# first solve to go to an appropriate orientation

#facing forward
p0 = [0.3,0.6,0.5]
p1 = vo.add(p0,[0,1,0])
p2 = vo.add(p0,[1,0,0])
#facing downward
# p0 = [0.2,0.6,0.5]
# p1 = vo.add(p0,[0,1,0])
# p2 = vo.add(p0,[0,0,-1])

# goal = ik.objective(robot.link(20),local=[[0,0,0],[0,1,0],[0,0,1]],world=[p0,p1,p2])
# if ik.solve_global(goal):
#     print("Hooray, IK solved")
#     print("Resulting config:",robot.getConfig())

# #fixed orientation goal
# goal = ik.fixed_rotation_objective(robot.link(20))

WSgrid = klampt.model.workspace.compute_workspace(robot.link(20), \
    goal, Nsamples=100000, resolution=0.05, self_collision = True, all_tests = True, fixed_links = [0,1,2,3,4,5])
#WSgrid = klampt.model.workspace.compute_workspace(robot.link(20), \
    # (0,0.0,0), Nsamples=100000, resolution=0.05, self_collision = True, all_tests = True, fixed_links = [0,1,2,3,4,5,6,7,8,9,10,11,12,21,22,23])
geom = Geometry3D(WSgrid['self_collision_free']) #'workspace'
geom2 = geom.convert('TriangleMesh',0.001)
vis.add('WS',geom2)

vis.show()              #open the window
t0 = time.time()
while vis.shown():
    time.sleep(0.01)    #loop is called ~100x times per second
vis.kill()    
