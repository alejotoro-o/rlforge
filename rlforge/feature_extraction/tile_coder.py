import numpy as np
from rlforge.feature_extraction import tiles3

class TileCoder:
    def __init__(self, dims_ranges, iht_size=4096, num_tilings=8, num_tiles=8, wrap_dims=()):
        """
        
        """
        self.iht = tiles3.IHT(iht_size)
        self.iht_size = iht_size
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

        self.scales = np.zeros(len(dims_ranges))

        for i, lims in enumerate(dims_ranges):
            self.scales[i] = self.num_tiles/(lims[1] - lims[0])

        self.wrap_widths = [self.num_tiles if wrap else False for wrap in wrap_dims]

    def get_tiles(self, x):
        """

        """
        scaled_input = list(x*self.scales)
                    
        if len(self.wrap_widths) > 0:
            active_tiles = tiles3.tileswrap(self.iht, self.num_tilings, scaled_input, wrapwidths=self.wrap_widths)
        else:
            active_tiles = tiles3.tiles(self.iht, self.num_tilings, scaled_input)
        
        return np.array(active_tiles)

