# basic imports
import numpy as np
import os
import pyqtgraph as qg
import cv2
# framework imports

from cobel.frontends.frontends_blender import FrontendBlenderInterface
from cobel.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import \
    ManualTopologyGraphNoRotation
from cobel.observations.image_observations import ImageObservationBaseline

os.environ['BLENDER_EXECUTABLE_PATH'] = "C:/Users/wafor/Desktop/blender-2.79b-windows64/blender.exe"

if __name__ == '__main__':
    demo_scene = "C:/Users/wafor/AppData/Local/Programs/Python/Python39/Lib/site-packages" + \
                 "/cobel/environments/environments_blender/linear_track_1x16.blend"

    path = os.path.dirname(os.path.abspath(__file__))
    info_path = os.path.split(demo_scene)[-1].split(".")[0]
    info_path = os.path.join(path, info_path)

    mainWindow = qg.GraphicsLayoutWidget(title='Demo: DQN')
    mainWindow.show()

    # a dictionary that contains all employed modules
    modules = dict()
    modules['world'] = FrontendBlenderInterface(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], mainWindow, True)
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(modules, {'startNodes': [0], 'goalNodes': [3],
                                                                                'cliqueSize': 2})
    modules['spatial_representation'].set_visual_debugging(True, mainWindow)

    # delete previous directory contents
    # os.system('rm -rf ' + info_path)
    # re-create the worldInfo directory
    os.makedirs(info_path, exist_ok=True)

    safeZoneDimensions = np.array([modules['world'].minX, modules['world'].minY, modules['world'].minZ,
                                   modules['world'].maxX, modules['world'].maxY, modules['world'].maxZ])
    np.save(info_path + '/safeZoneDimensions.npy', safeZoneDimensions)
    np.save(info_path + '/safeZonePolygon.npy', modules['world'].safeZonePolygon)
    np.save(info_path + '/safeZoneVertices.npy', modules['world'].safeZoneVertices)
    np.save(info_path + '/safeZoneSegments.npy', modules['world'].safeZoneSegments)

    # store environment information
    nodes = np.array(modules['world'].getManuallyDefinedTopologyNodes())
    nodes = nodes[nodes[:, 0].argsort()]
    edges = np.array(modules['world'].getManuallyDefinedTopologyEdges())
    edges = edges[edges[:, 0].argsort()]
    np.save(info_path + '/topologyNodes.npy', nodes)
    np.save(info_path + '/topologyEdges.npy', edges)

    # store referenceImages sampled images from contexts A and B

    # Note: context A is conditioning and ROF (both red light), context b is extinction (white light). The light colors can be changed on demand.

    # prepare context A
    # switch on white light

    imageDims = (30, 1)
    referenceImages = []

    for ni in range(len(modules['spatial_representation'].nodes)):
        node = modules['spatial_representation'].nodes[ni]
        # only for valid nodes, the 'NoneNode' is not considered here
        if node.index != -1:
            # propel the simulation
            modules[
                'spatial_representation'].nextNode = node.index  # required by the WORLD_ABAFromImagesInterface
            modules['world'].step_simulation_without_physics(node.x, node.y, 90.0)
            # the observation is plainly the robot's camera image data
            observation = modules['world'].envData['imageData']
            # for now, cut out a single line from the center of the image (w.r.t. height) and use this as an observation in order
            # to save computational resources
            # observation=observation[29:30,:,:]
            # scale the one-line image to further reduce computational demands
            observation = cv2.resize(observation, dsize=(imageDims))
            # observation=np.flip(observation,0)
            # cv2.imshow('Test',observation)
            # cv2.waitKey(0)
            referenceImages += [observation]

    images = np.array(referenceImages)
    np.save(info_path + '/images.npy', images)
    modules['world'].stopBlender()
    # and also stop visualization
    mainWindow.close()
