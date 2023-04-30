import astra
import numpy as np

class ForwardOperator:
    def __init__(self):
        # Distances and pixel size in mm
        dist_src_center = 410.66
        dist_src_detector = 553.74
        pixelsize = 0.2 * dist_src_center / dist_src_detector
        num_detectors = 560
        vol_geom_id = astra.create_vol_geom(512, 512)
        angles = np.linspace(0, 2 * np.pi, 721)

        projection_id = astra.create_proj_geom(
            "fanflat",
            dist_src_detector / dist_src_center,
            num_detectors,
            angles,
            dist_src_center / pixelsize,
            (dist_src_detector - dist_src_center) / pixelsize)

        projector_id = astra.create_projector("cuda", projection_id, vol_geom_id)

        volume_id = astra.data2d.create("-vol", vol_geom_id)

        self.volume_id = volume_id
        self.projector_id = projector_id

    def __call__(self, image):
        astra.data2d.store(self.volume_id, image)
        sinogram_id, sinogram = astra.create_sino(self.volume_id, self.projector_id)
        # Release memory
        astra.data2d.delete(sinogram_id)
        return sinogram
