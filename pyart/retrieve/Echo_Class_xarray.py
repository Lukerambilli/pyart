import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.ndimage import uniform_filter
from scipy.ndimage import convolve # Explicitly import convolve

# Custom colormap for reflectivity, if not available in your environment, you might need to define it or use a standard one.
# For example, if NWSRef is not defined, you can use 'viridis' or 'jet'.
# If you have Py-ART installed, NWSRef should be available.
try:
    import pyart.graph.cm_colorblind as cm_colorblind
    plt.colormaps.register(cm_colorblind.NWSRef)
except ImportError:
    print("Py-ART not found. NWSRef colormap might not be available. Using 'viridis' as fallback for reflectivity.")
    # You can define a simple reflectivity colormap if Py-ART is not installed
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["white", "cyan", "deepskyblue", "lime", "green", "yellow", "gold", "orange", "red", "firebrick", "darkred", "purple", "indigo", "black"]
    n_bins = [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5] # Number of bins per color segment
    cmap_name = "NWSRef_fallback"
    NWSRef_fallback = LinearSegmentedColormap.from_list(cmap_name, colors, N=sum(n_bins))
    plt.colormaps.register(NWSRef_fallback)
    cmap_dbz_name = 'NWSRef_fallback'
else:
    cmap_dbz_name = 'NWSRef'


class ConvectiveStratiformClassifier:
    """
    Classifies precipitation into convective and stratiform types using 
    the Steiner et al. (1995) algorithm with optimized vectorized operations.
    """
    
    def __init__(self, intense=42.0, peak_relation="default", area_relation="medium", 
                 bkg_rad=11000.0, use_intense=True):
        """
        Initialize the classifier with Steiner algorithm parameters.
        
        Parameters:
        -----------
        intense : float
            Intensity threshold in dBZ for automatic convective classification
        peak_relation : str
            Peakedness relation type ("default" or "sgp")
        area_relation : str  
            Area relation type ("small", "medium", "large", "sgp")
        bkg_rad : float
            Background radius in meters
        use_intense : bool
            Whether to use intensity threshold criterion
        """
        self.intense = intense
        self.peak_relation = peak_relation
        self.area_relation = area_relation
        self.bkg_rad = bkg_rad
        self.use_intense = use_intense
        
        # Mapping dictionaries for relations
        self.area_rel_map = {"small": 0, "medium": 1, "large": 2, "sgp": 3}
        self.peak_rel_map = {"default": 0, "sgp": 1}
    
    def _convective_radius(self, ze_bkg):
        """
        Determine convective radius based on background reflectivity.
        Vectorized version supporting arrays.
        """
        area_relation = self.area_rel_map[self.area_relation]
        
        # Initialize output array
        conv_rad = np.zeros_like(ze_bkg)
        
        if area_relation == 0:  # small
            conv_rad = np.where(ze_bkg < 30, 1000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 30) & (ze_bkg < 35), 2000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 35) & (ze_bkg < 40), 3000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 40) & (ze_bkg < 45), 4000.0, conv_rad)
            conv_rad = np.where(ze_bkg >= 45, 5000.0, conv_rad)
            
        elif area_relation == 1:  # medium
            conv_rad = np.where(ze_bkg < 25, 1000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 25) & (ze_bkg < 30), 2000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 30) & (ze_bkg < 35), 3000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 35) & (ze_bkg < 40), 4000.0, conv_rad)
            conv_rad = np.where(ze_bkg >= 40, 5000.0, conv_rad)
            
        elif area_relation == 2:  # large
            conv_rad = np.where(ze_bkg < 20, 1000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 20) & (ze_bkg < 25), 2000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 25) & (ze_bkg < 30), 3000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 30) & (ze_bkg < 35), 4000.0, conv_rad)
            conv_rad = np.where(ze_bkg >= 35, 5000.0, conv_rad)
            
        elif area_relation == 3:  # sgp
            conv_rad = np.where(ze_bkg < 40, 0.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 40) & (ze_bkg < 45), 1000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 45) & (ze_bkg < 50), 2000.0, conv_rad)
            conv_rad = np.where((ze_bkg >= 50) & (ze_bkg < 55), 6000.0, conv_rad)
            conv_rad = np.where(ze_bkg >= 55, 8000.0, conv_rad)
            
        return conv_rad
    
    def _peakedness_threshold(self, ze_bkg):
        """
        Calculate peakedness threshold based on background reflectivity.
        Vectorized version supporting arrays.
        """
        peak_relation = self.peak_rel_map[self.peak_relation]
        
        if peak_relation == 0:  # default
            peak = np.where(ze_bkg < 0, 10.0, 
                            np.where(ze_bkg < 42.43, 10.0 - ze_bkg**2 / 180.0, 0.0))
        elif peak_relation == 1:  # sgp
            peak = np.where(ze_bkg < 0, 14.0,
                            np.where(ze_bkg < 42.43, 14.0 - ze_bkg**2 / 180.0, 4.0))
        
        return peak
    
    def _calculate_background_reflectivity(self, refl, dx, dy):
        """
        Calculate background reflectivity using efficient convolution approach.
        This is much faster than nested loops.
        """
        # Create circular kernel for background calculation
        bkg_rad_x = int(self.bkg_rad / dx)
        bkg_rad_y = int(self.bkg_rad / dy)
        
        # Create coordinate grids for kernel
        y_kernel, x_kernel = np.ogrid[-bkg_rad_y:bkg_rad_y+1, -bkg_rad_x:bkg_rad_x+1]
        
        # Create circular mask
        kernel_mask = (x_kernel * dx)**2 + (y_kernel * dy)**2 <= self.bkg_rad**2
        
        # Convert reflectivity to linear units for averaging
        refl_linear = np.where(np.isfinite(refl) & (refl > 0), 
                               10**(refl/10.0), np.nan)
        
        # Use uniform_filter for efficient convolution-like operation
        # We need to handle NaN values properly
        valid_mask = np.isfinite(refl_linear)
        
        # Calculate sum and count using convolve
        
        # Sum of linear reflectivity values
        sum_linear = convolve(np.where(valid_mask, refl_linear, 0), 
                              kernel_mask.astype(float), mode='constant', cval=0)
        
        # Count of valid values  
        count_valid = convolve(valid_mask.astype(float), 
                               kernel_mask.astype(float), mode='constant', cval=0)
        
        # Calculate mean and convert back to dBZ
        mean_linear = np.where(count_valid > 0, sum_linear / count_valid, np.nan)
        ze_bkg = np.where(np.isfinite(mean_linear) & (mean_linear > 0),
                          10 * np.log10(mean_linear), 0)
        
        return ze_bkg
    
    def classify_precipitation(self, dbz_2d, dx, dy):
        """
        Classify 2D reflectivity data using the Steiner et al. (1995) algorithm.
        Optimized version with minimal loops.
        
        Parameters:
        -----------
        dbz_2d : numpy.ndarray
            2D reflectivity data in dBZ
        dx, dy : float
            Grid spacing in x and y directions (meters)
            
        Returns:
        --------
        classification : numpy.ndarray
            Classification array: 0=undefined, 1=stratiform, 2=convective  
        """
        rows, cols = dbz_2d.shape
        classification = np.zeros((rows, cols), dtype=int)
        
        # Valid precipitation mask (at least for initial checks)
        valid_precip = np.isfinite(dbz_2d) & (dbz_2d > 0)
        
        # Step 1: Calculate background reflectivity for all points
        ze_bkg = self._calculate_background_reflectivity(dbz_2d, dx, dy)
        
        # Step 2: Intensity-based convective classification
        convective_mask = np.zeros_like(dbz_2d, dtype=bool)
        
        if self.use_intense:
            intense_mask = valid_precip & (dbz_2d >= self.intense)
            convective_mask |= intense_mask
        
        # Step 3: Peakedness-based convective classification
        peak_threshold = self._peakedness_threshold(ze_bkg)
        peakedness_mask = (valid_precip & 
                           (ze_bkg > 0) & 
                           ((dbz_2d - ze_bkg) >= peak_threshold) &
                           ~convective_mask)  # Don't double-classify intense points
        
        convective_mask |= peakedness_mask
        
        # Step 4: Expand convective areas
        expanded_convective = self._expand_convective_areas(
            convective_mask, ze_bkg, dx, dy, dbz_2d)
        
        # Step 5: Final classification
        classification[expanded_convective & valid_precip] = 2  # Convective
        classification[valid_precip & ~expanded_convective] = 1  # Stratiform
        
        return classification
    
    def _expand_convective_areas(self, convective_centers, ze_bkg, dx, dy, refl):
        """
        Expand convective areas based on convective radius.
        This requires some iteration but is optimized.
        """
        expanded = convective_centers.copy()
        rows, cols = convective_centers.shape
        
        # Get convective radius for all points
        conv_rad = self._convective_radius(ze_bkg)
        
        # Find all convective center points
        conv_y, conv_x = np.where(convective_centers)
        
        # For each convective center, expand the area
        for cy, cx in zip(conv_y, conv_x):
            if conv_rad[cy, cx] > 0:
                # Calculate expansion radius in pixels
                rad_x = int(conv_rad[cy, cx] / dx)
                rad_y = int(conv_rad[cy, cx] / dy)
                
                # Define bounds
                x_min = max(0, cx - rad_x)
                x_max = min(cols, cx + rad_x + 1)
                y_min = max(0, cy - rad_y)
                y_max = min(rows, cy + rad_y + 1)
                
                # Create coordinate arrays for the sub-region
                x_coords = np.arange(x_min, x_max)
                y_coords = np.arange(y_min, y_max)
                xx, yy = np.meshgrid(x_coords, y_coords)
                
                # Calculate distances
                distances = np.sqrt((xx - cx)**2 * dx**2 + (yy - cy)**2 * dy**2)
                
                # Mark points within convective radius as convective
                within_radius = distances <= conv_rad[cy, cx]
                valid_points = np.isfinite(refl[y_min:y_max, x_min:x_max]) & (refl[y_min:y_max, x_min:x_max] > 0)
                
                expanded[y_min:y_max, x_min:x_max][within_radius & valid_points] = True
        
        return expanded
    
    def classify_from_dataset(self, dbz_da):
        """
        Classify precipitation from xarray DataArray.
        
        Parameters:
        -----------
        dbz_da : xarray.DataArray
            Reflectivity data array with dimensions (y, x) or (time, z, y, x) etc.
            Assumes it's already sliced to 2D (e.g., specific time and altitude).
            
        Returns:
        --------
        classification : numpy.ndarray
            Classification array
        """
        # Ensure dbz_da is a 2D array. If it has extra dimensions, select the first.
        # This part might need adjustment depending on the exact structure of your xarray.
        # For simplicity, assuming it's already 2D or we take the first available 2D slice.
        if dbz_da.ndim == 4:
            dbz_2d = dbz_da.values[0, 0, :, :] # Assuming (time, z, y, x)
        elif dbz_da.ndim == 3:
            dbz_2d = dbz_da.values[0, :, :] # Assuming (time, y, x) or (z, y, x)
        elif dbz_da.ndim == 2:
            dbz_2d = dbz_da.values # Already 2D
        else:
            raise ValueError("dbz_da must be a 2D, 3D or 4D xarray DataArray.")

        # Extract dx and dy from coordinates if available, otherwise assume fixed.
        # This is a crucial assumption. If your xarray does not have 'x' and 'y' coords
        # or they are not uniform, you'll need to pass dx, dy explicitly.
        # For this example, assuming 'x' and 'y' coordinates are spatial.
        try:
            # Assuming x and y are evenly spaced
            dx = float(np.abs(dbz_da.x.values[1] - dbz_da.x.values[0]))
            dy = float(np.abs(dbz_da.y.values[1] - dbz_da.y.values[0]))
        except (AttributeError, IndexError):
            print("Warning: Could not determine dx, dy from xarray coordinates. Assuming 1000m for both.")
            dx = 1000.0 # Default value if coords are not setup
            dy = 1000.0 # Default value if coords are not setup

        return self.classify_precipitation(dbz_2d, dx, dy)


def create_circular_mask(x_coords, y_coords, radius_km=250):
    """
    Create a circular mask of radius_km centered on the grid (0,0).
    Assumes x_coords and y_coords are in meters.
    """
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    distances = np.sqrt(x_grid**2 + y_grid**2) / 1000  # Convert meters to km
    return distances <= radius_km


def plot_cloud_classification(dbz_da):
    """
    Plot convective/stratiform classification and reflectivity within 250 km circle.
    
    Parameters:
    -----------
    dbz_da : xarray.DataArray
        Reflectivity data array. Assumes it has 'x', 'y' (in meters), 'lat', 'lon'
        coordinates and potentially 'time', 'z'.
    """
    
    # Extract 2D reflectivity, and coordinates
    # This expects dbz_da to be at least 2D (y, x) or have dimensions that can be broadcasted.
    # We should explicitly handle the dimensions if it's 3D/4D
    if dbz_da.ndim == 4:
        dbz_2d = dbz_da.isel(time=0, z=0).values # Taking the first time and altitude slice
    elif dbz_da.ndim == 3:
        # Assuming (time, y, x) or (z, y, x). Take the first slice.
        dbz_2d = dbz_da.isel({dbz_da.dims[0]: 0}).values
    elif dbz_da.ndim == 2:
        dbz_2d = dbz_da.values
    else:
        raise ValueError("dbz_da must be a 2D, 3D or 4D xarray DataArray.")

    # Get grid spacing from xarray coordinates
    try:
        dx = float(np.abs(dbz_da.x.values[1] - dbz_da.x.values[0]))
        dy = float(np.abs(dbz_da.y.values[1] - dbz_da.y.values[0]))
    except (AttributeError, IndexError):
        print("Warning: Could not determine dx, dy from xarray coordinates. Using 1000m for both.")
        dx = 1000.0 # Fallback
        dy = 1000.0 # Fallback

    # Create circular mask (250 km radius) using the 'x' and 'y' coordinates from the DataArray
    # Assuming x and y coordinates are in meters and centered around 0.
    mask = create_circular_mask(dbz_da.x.values, dbz_da.y.values, radius_km=250)
    
    # Initialize classifier with Steiner parameters
    classifier = ConvectiveStratiformClassifier(
        intense=42.0,
        peak_relation="default",  
        area_relation="medium",
        bkg_rad=11000.0,
        use_intense=True
    )
    
    # Get classification
    # Pass the 2D numpy array and dx, dy to the classifier's core method
    classification = classifier.classify_precipitation(dbz_2d, dx, dy)
    
    # Extract lat, and lon for plotting extent
    lat = dbz_da.lat.values
    lon = dbz_da.lon.values
    
    # Get time and altitude for title (handle if not present)
    # time_val = dbz_da.time.values[0] if 'time' in dbz_da.coords else (
    # dbz_da.time.values[0] if 'time' in dbz_da.dims else "N/A")
    
    if 'time' in dbz_da.coords or 'time' in dbz_da.dims:
        time_val = pd.to_datetime(dbz_da.time.values).strftime('%Y-%m-%d %H:%M:%S')
    else:
        time_val = "N/A"

    altitude_val = dbz_da.z.values[0] if 'z' in dbz_da.dims else "N/A"
    
    # Apply circular mask for plotting
    classification_masked = np.where(mask, classification, np.nan)
    dbz_masked = np.where(mask, dbz_2d, np.nan)
    
    # --- IMPORTANT FIX: Calculate statistics ONLY within the masked region ---
    
    # Filter for valid precipitation within the mask for statistical calculations
    # `classification_masked` and `dbz_masked` already have NaNs outside the circle.
    # We now filter for valid precipitation *within* the circle for statistics.
    valid_precip_in_circle = ~np.isnan(dbz_masked) & (dbz_masked > 0)
    
    # Calculate statistics within the circular mask
    total_pixels_in_circle = np.sum(mask) # Count of pixels inside the circle
    
    # Use the *masked* classification and reflectivity for statistics
    convective_pixels = np.sum(classification_masked == 2)
    stratiform_pixels = np.sum(classification_masked == 1)
    
    # Pixels where classification is 0 (undefined/non-precip) AND are within the circle
    # And specifically where original dbz was also not valid precip
    non_precip_pixels_in_circle = np.sum(mask & ((dbz_2d <= 0) | np.isnan(dbz_2d)))
    
    # Count of valid precipitation pixels within the circle (where dbz_2d > 0 and not NaN)
    valid_precip_pixels_in_circle = np.sum(valid_precip_in_circle)
    
    # Calculate percentages relative to the *total pixels within the circle*
    non_precip_pct = (non_precip_pixels_in_circle / total_pixels_in_circle * 100) if total_pixels_in_circle > 0 else 0
    
    # Percentage of valid precipitation *within the circular region*
    valid_precip_pct_of_circle = (valid_precip_pixels_in_circle / total_pixels_in_circle * 100) if total_pixels_in_circle > 0 else 0
    
    # Convective and Stratiform percentages *of the valid precipitation within the circle*
    convective_pct_of_valid_precip = (convective_pixels / valid_precip_pixels_in_circle * valid_precip_pct_of_circle) if valid_precip_pixels_in_circle > 0 else 0
    stratiform_pct_of_valid_precip = (stratiform_pixels / valid_precip_pixels_in_circle * valid_precip_pct_of_circle) if valid_precip_pixels_in_circle > 0 else 0

    # Reflectivity statistics within circle
    dbz_values_for_stats = dbz_masked[valid_precip_in_circle]
    if len(dbz_values_for_stats) > 0:
        mean_dbz = np.mean(dbz_values_for_stats)
        max_dbz = np.max(dbz_values_for_stats)
        min_dbz = np.min(dbz_values_for_stats) # Although not displayed, good to have.
    else:
        mean_dbz = max_dbz = min_dbz = 0
    
    # Print comprehensive statistics
    print("\n=== STEINER CLASSIFICATION STATISTICS WITHIN 250 KM CIRCLE ===")
    print(f"Total pixels within 250km circle: {total_pixels_in_circle}")
    print(f"cloud void pixels within circle: {non_precip_pixels_in_circle} ({non_precip_pct:.2f}%)")
    print(f"cloud pixels within circle: {valid_precip_pixels_in_circle} ({valid_precip_pct_of_circle:.2f}% of total circle)")
    print(f"Convective cloud pixels: {convective_pixels} ({convective_pct_of_valid_precip:.2f}% of valid precip)")
    print(f"Stratiform cloud pixels: {stratiform_pixels} ({stratiform_pct_of_valid_precip:.2f}% of valid precip)")
    print(f"Mean reflectivity: {mean_dbz:.2f} dBZ")
    print(f"Max reflectivity: {max_dbz:.2f} dBZ")
    
    # Create plots with adjusted figure size and spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Classification
    cmap_class = plt.cm.colors.ListedColormap(['white', 'orange', 'green'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm_class = plt.cm.colors.BoundaryNorm(bounds, cmap_class.N)
    
    im1 = ax1.imshow(classification_masked, cmap=cmap_class, norm=norm_class, origin='lower',
                     extent=[lon.min(), lon.max(), lat.min(), lat.max()])
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2], shrink=0.7, aspect=20, pad=0.05,
                         format=plt.FuncFormatter(lambda x, _: ['None', 'Stratiform', 'Convective'][int(x)]))
    cbar1.set_label('Cloud Type', fontweight='bold', fontsize=11, labelpad=  15)
    cbar1.ax.tick_params(labelsize=10)
    
    # Make colorbar tick labels bold, rotated, and shifted upward
    labels = ['None', 'Stratiform', 'Convective']
    for i, label in enumerate(cbar1.ax.get_yticklabels()):
        label.set_fontweight('bold')
        label.set_rotation(90)
        # Set vertical alignment to 'bottom' and apply a small upward transform
        label.set_verticalalignment('center')
        label.set_y(label.get_position()[1] + 0.01)  # Increase offset for more noticeable shift


    
    # Add statistics text
    # Display percentages relative to the total area *within the circle*
    stats_text = (f'Cloud Void: {non_precip_pct:.1f}%\n'
                  f'Cloud Cover: {valid_precip_pct_of_circle:.1f}%\n'
                  f'Conv: {convective_pct_of_valid_precip:.1f}%\n'
                  f'Strat: {stratiform_pct_of_valid_precip:.1f}%')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_title(f'Time: {time_val}', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Longitude (°E)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Latitude (°N)', fontweight='bold', fontsize=11)
    ax1.tick_params(labelsize=10)
    
    # Make tick labels bold
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    ax1.set_aspect('equal')
    
    # Plot 2: Reflectivity
    im2 = ax2.imshow(dbz_masked, cmap=cmap_dbz_name, origin='lower', vmin=0, vmax=60,
                     extent=[lon.min(), lon.max(), lat.min(), lat.max()])
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=20, pad=0.05)
    cbar2.set_label('Reflectivity (dBZ)', fontweight='bold', fontsize=11)
    cbar2.ax.tick_params(labelsize=10)
    
    # Make colorbar tick labels bold
    for label in cbar2.ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Add reflectivity statistics
    dbz_stats_text = f'Mean: {mean_dbz:.1f} dBZ\nMax: {max_dbz:.1f} dBZ'
    ax2.text(0.02, 0.98, dbz_stats_text, 
             transform=ax2.transAxes, fontsize=9, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax2.set_title(f'Time: {time_val}', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Latitude (°N)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Longitude (°E)', fontweight='bold', fontsize=11)
    ax2.tick_params(labelsize=10)
    
    # Make tick labels bold
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontweight('bold')
    
    ax2.set_aspect('equal')
    
    # Adjust layout to prevent colorbar overlap
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.12, wspace = 0.15)
    plt.show()
    
    return {
        'total_pixels_in_circle': total_pixels_in_circle,
        'valid_precip_pixels_in_circle': valid_precip_pixels_in_circle,
        'convective_pixels': convective_pixels,
        'stratiform_pixels': stratiform_pixels,
        'non_precip_pct': non_precip_pct,
        'valid_precip_pct_of_circle': valid_precip_pct_of_circle,
        'convective_pct_of_valid_precip': convective_pct_of_valid_precip,
        'stratiform_pct_of_valid_precip': stratiform_pct_of_valid_precip,
        'mean_dbz': mean_dbz,
        'max_dbz': max_dbz
    }