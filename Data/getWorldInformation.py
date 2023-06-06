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
    # Change paths and don't forget to change the goal/start nodes!!!
    demo_scene = "C:/Users/wafor/Desktop/cobel_saves/environments_blender/linear_track_1x16.blend"
    output_path = "C:/Users/wafor/Desktop/linear_track_16"

    mainWindow = qg.GraphicsLayoutWidget(title='Demo: DQN')
    mainWindow.show()

    # a dictionary that contains all employed modules
    modules = dict()
    modules['world'] = FrontendBlenderInterface(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], mainWindow, True)
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(modules, {'start_nodes': [0], 'goal_nodes': [15],
                                                                                'clique_size': 2})
    modules['spatial_representation'].set_visual_debugging(True, mainWindow)

    # delete previous directory contents
    # os.system('rm -rf ' + info_path)
    # re-create the worldInfo directory
    os.makedirs(output_path, exist_ok=True)

    safeZoneDimensions = np.array([modules['world'].min_x, modules['world'].min_y, modules['world'].min_z,
                                   modules['world'].max_x, modules['world'].max_y, modules['world'].max_z])
    np.save(output_path + '/safeZoneDimensions.npy', safeZoneDimensions)
    np.save(output_path + '/safeZonePolygon.npy', modules['world'].safe_zone_polygon)
    np.save(output_path + '/safeZoneVertices.npy', modules['world'].safe_zone_vertices)
    np.save(output_path + '/safeZoneSegments.npy', modules['world'].safe_zone_segments)

    # store environment information
    nodes = np.array(modules['world'].get_manually_defined_topology_nodes())
    nodes = nodes[nodes[:, 0].argsort()]
    edges = np.array(modules['world'].get_manually_defined_topology_edges())
    edges = edges[edges[:, 0].argsort()]
    np.save(output_path + '/topologyNodes.npy', nodes)
    np.save(output_path + '/topologyEdges.npy', edges)

    # store referenceImages sampled images from contexts A and B

    # Note: context A is conditioning and ROF (both red light), context b is extinction (white light). The light colors can be changed on demand.

    # prepare context A
    # switch on white light

    image_dims = (30, 1)
    reference_images = []

    for ni in range(len(modules['spatial_representation'].nodes)):
        node = modules['spatial_representation'].nodes[ni]
        # only for valid nodes, the 'NoneNode' is not considered here
        if node.index != -1:
            # propel the simulation
            modules[
                'spatial_representation'].next_node = node.index  # required by the WORLD_ABAFromImagesInterface
            modules['world'].step_simulation_without_physics(node.x, node.y, 90.0)
            # the observation is plainly the robot's camera image data
            observation = modules['world'].env_data['image']
            # for now, cut out a single line from the center of the image (w.r.t. height) and use this as an observation in order
            # to save computational resources
            # observation=observation[29:30,:,:]
            # scale the one-line image to further reduce computational demands
            observation = cv2.resize(observation, dsize=(image_dims))
            # observation=np.flip(observation,0)
            # cv2.imshow('Test',observation)
            # cv2.waitKey(0)
            reference_images += [observation]

    images = np.array(reference_images)
    np.save(output_path + '/images.npy', images)
    modules['world'].stop_blender()
    # and also stop visualization
    mainWindow.close()
