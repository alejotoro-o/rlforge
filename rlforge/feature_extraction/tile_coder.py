import numpy as np
import tiles3

class TileCoder:
    """
    Tile coding feature representation for reinforcement learning.

    Tile coding is a form of coarse coding that maps continuous input
    variables into a sparse binary feature vector. It is widely used in
    reinforcement learning to approximate value functions efficiently.

    This implementation uses the `tiles3` module to generate active tiles
    for given inputs. Multiple tilings are overlaid to provide finer
    resolution and generalization.

    Parameters
    ----------
    dims_ranges : list of tuple(float, float)
        List of ranges for each input dimension, e.g. [(min_x, max_x), (min_y, max_y)].
    iht_size : int, optional
        Size of the index hash table (default: 4096).
    num_tilings : int, optional
        Number of tilings (default: 8).
    num_tiles : int, optional
        Number of tiles per dimension per tiling (default: 8).
    wrap_dims : tuple of bool, optional
        Flags indicating whether each dimension should wrap around
        (useful for periodic variables like angles). Default is ().

    Attributes
    ----------
    iht : tiles3.IHT
        Index hash table used by the tile coding implementation.
    scales : numpy.ndarray
        Scaling factors for each input dimension to map values into tile indices.
    wrap_widths : list
        Wrap widths for each dimension, or False if no wrapping is applied.
    """

    def __init__(self, dims_ranges, iht_size=4096, num_tilings=8, num_tiles=8, wrap_dims=()):
        """
        Initialize the tile coder with the given input ranges and parameters.
        """
        self.iht = tiles3.IHT(iht_size)
        self.iht_size = iht_size
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

        self.scales = np.zeros(len(dims_ranges))
        for i, lims in enumerate(dims_ranges):
            self.scales[i] = self.num_tiles / (lims[1] - lims[0])

        self.wrap_widths = [self.num_tiles if wrap else False for wrap in wrap_dims]

    def get_tiles(self, x):
        """
        Compute active tile indices for a given input vector.

        Parameters
        ----------
        x : array-like
            Input vector of continuous values, one per dimension.

        Returns
        -------
        numpy.ndarray
            Array of active tile indices corresponding to the input.

        Notes
        -----
        - Inputs are scaled according to the dimension ranges provided at initialization.
        - If `wrap_dims` was specified, periodic wrapping is applied to the corresponding dimensions.
        - The returned indices can be used as features in linear function approximation.
        """
        scaled_input = list(x * self.scales)

        if len(self.wrap_widths) > 0:
            active_tiles = tiles3.tileswrap(
                self.iht, self.num_tilings, scaled_input, wrapwidths=self.wrap_widths
            )
        else:
            active_tiles = tiles3.tiles(self.iht, self.num_tilings, scaled_input)

        return np.array(active_tiles)