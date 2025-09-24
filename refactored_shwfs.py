#%%
"""
Refactored SHWFS reconstruction using class-based approach for better organization,
testability, and maintainability.
20250915 warrenbfoster
"""
import os
import sys
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path
import pickle

current = Path(__file__)
for path in [current] + list(current.parents):
    if (path / 'mirror_control').exists():
        workspace_root = str(path)
        break

if workspace_root and workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


from high_level_SH_utils import xyr_pupil_definition
from SH_utils import SouthwellIntegration, define_reference, get_quiver, prepare_image, quiver2regular_slope, trim_maps_to_square
# Import functions from mirror_control/shared
from mirror_control.shared.zernike_utils import get_M_and_C, remove_modes
from mirror_control.shared.General_zernike_matrix import General_zernike_matrix

@dataclass
class SHWFSConfig:
    """Configuration parameters for SHWFS reconstruction."""
    # Calibration parameters
    star_separation_pixels: float = 5.63116 # distance between 2 stars in pixels
    star_separation_arcsec: float = 5.5 # angular separation of 2 stars in arcseconds
    star_separation_rad: float = star_separation_arcsec * 4.848e-6  # angular separation in radians
    slopeMagnification: float = 1.0 #Calibration factor for slope values

    mirror_diameter_mm: float = 30 * 25.4
    mirror_inner_diameter_mm: float = 6 * 25.4
    lenslet_diameter_mm: float = 0.3
    mirror_focal_length_mm: float = 2534.3
    reimaging_lens_focal_length_mm: float = 25
    integration_step_mm: float = 3
    
    # Processing parameters
    zernike_modes: int = 44
    remove_modes: List[int] = None
    percentile_threshold: float = 0.995
    pv_outlier_fraction: float = 0.001
    
    # Output parameters
    default_output_plots: bool = False
    save_intermediate_results: bool = False
    default_verbose: bool = True
    
    def __post_init__(self):
        if self.remove_modes is None:
            self.remove_modes = [0, 1, 2, 4]  # Piston, tip, tilt, focus


@dataclass
class ReconstructionResult:
    """Results from SHWFS reconstruction."""
    surface_um: np.ndarray
    mean_surface: np.ndarray
    rms_nm: float
    pv_nm: float
    zernike_coefficients: Optional[np.ndarray] = None
    processing_time: float = 0.0


class SHWFSReconstructor:
    """
    Shack-Hartmann Wavefront Sensor reconstruction processor.
    
    Separates concerns into distinct phases:
    1. Calibration/Setup
    2. Image Processing 
    3. Wavefront Reconstruction
    4. Analysis/Output
    """
    
    def __init__(self, config: SHWFSConfig):
        self.config = config
        self._calibration_data = None
        self._pupil_definition = None
        self._output_plots = config.default_output_plots  # Initialize from config
        self._verbose = config.default_verbose  # Initialize from config
        
    def set_output_plots(self, enable: bool) -> None:
        """Enable or disable intermediate plot output for all subsequent operations."""
        self._output_plots = enable
        
    def get_output_plots(self) -> bool:
        """Get current output plots setting."""
        return self._output_plots
        
    def disable_all_output(self) -> None:
        """Disable all intermediate output for performance."""
        self._output_plots = False
        self._verbose = False
        
    def calibrate(self, sh_path: Path, folder_path: Path, 
                  redefine_pupil: bool = False) -> None:
        """Perform initial calibration and setup."""
        # Define pupil location
        self._pupil_definition = self._define_pupil(folder_path, sh_path, redefine_pupil)
        
        # Define lenslet grid
        self._calibration_data = self._define_lenslets(
            folder_path, sh_path, self._pupil_definition, redefine_pupil
        )
        
    def reconstruct_from_folder(self, folder_path: Path, 
                               subset_list: Optional[List[str]] = None,
                               output_plots: Optional[bool] = None) -> ReconstructionResult:
        """Main reconstruction pipeline from folder of images."""
        if not self._is_calibrated():
            raise RuntimeError("Must call calibrate() before reconstruction")
            
        # Use method parameter if provided, otherwise use instance setting
        use_plots = self._should_output(output_plots)
            
        # Process images
        if subset_list is None:
            surfaces = self._process_all_images(folder_path, use_plots)
        else:
            surfaces = self._process_subset_images(folder_path, subset_list, use_plots)
            
        # Analyze results
        return self._analyze_surfaces(surfaces, folder_path)
        
    def reconstruct_single_image(self, image_path: Path,
                                output_plots: Optional[bool] = None) -> np.ndarray:
        """Reconstruct wavefront from single image."""
        if not self._is_calibrated():
            raise RuntimeError("Must call calibrate() before reconstruction")
            
        # Use method parameter if provided, otherwise use instance setting
        use_plots = self._should_output(output_plots)
            
        image, _ = self._prepare_image(image_path)
        return self._compute_wavefront_from_image(image, use_plots)
        
    def _is_calibrated(self) -> bool:
        """Check if calibration has been performed."""
        return (self._calibration_data is not None and 
                self._pupil_definition is not None)
    
    def _define_pupil(self, folder_path: Path, sh_path: Path, 
                     redefine: bool) -> dict:
        """Define pupil location within SH image."""
        
        # Path to cached pupil definition
        xyr_file_path = sh_path / 'xyr.pkl'
        
        # Try to load existing pupil definition if file exists and redefine=False
        if not redefine and os.path.isfile(str(xyr_file_path)):
            try:
                with open(xyr_file_path, 'rb') as f:
                    xyr = pickle.load(f)
                
                # Convert xyr array to structured dictionary
                pupil_def = {
                    'center_x': float(xyr[0]),
                    'center_y': float(xyr[1]), 
                    'radius': float(xyr[2]),
                    'xyr': xyr,  # Keep original format for compatibility
                    'source': 'loaded_from_cache'
                }
                
                return pupil_def
                
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                print(f"Warning: Could not load xyr.pkl file: {e}")
                print("Proceeding with pupil redefinition...")
        
        # If no cached file or redefine=True, compute new pupil definition
        # Call original function to get xyr and reference image
        xyr, jup_image = xyr_pupil_definition(str(folder_path), str(sh_path), redefine)
        
        # Convert to structured dictionary format
        pupil_def = {
            'center_x': float(xyr[0]),
            'center_y': float(xyr[1]),
            'radius': float(xyr[2]), 
            'xyr': xyr,  # Keep original format for compatibility
            'reference_image': jup_image,
            'source': 'newly_computed'
        }
        
        return pupil_def
        
    def get_xyr(self) -> np.ndarray:
        """Get pupil definition in original xyr array format for backward compatibility."""
        if not self._is_calibrated():
            raise RuntimeError("Must call calibrate() before accessing pupil data")
        return self._pupil_definition['xyr']
        
    def _should_output(self, output_plots: Optional[bool] = None) -> bool:
        """Determine if output should be generated based on settings hierarchy."""
        if output_plots is not None:
            return output_plots  # Method-level override
        return self._output_plots  # Instance-level setting
        
    def _define_lenslets(self, folder_path: Path, sh_path: Path, 
                        pupil_def: dict, redefine: bool) -> dict:
        """Define lenslet grid parameters."""
        # Implementation moved from lenslet_definition()
        if type(folder_path) is str:
            folder_path = Path(folder_path)
        reference_path = Path.joinpath(folder_path, 'references.pkl')
        if os.path.isfile(reference_path) and not redefine:
            try:
                with open(reference_path, 'rb') as f:
                    references = pickle.load(f)
                    return references
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                print(f"Warning: Could not load references.pkl file: {e}")
                print("Proceeding with lenslet redefinition...")

        referenceX, referenceY, magnification, nominalSpot, rotation = define_reference(folder_path, self._pupil_definition['xyr'], self._output_plots)
        references = {"refX":referenceX,
                    "refY":referenceY,
                    "magnification":magnification,
                    "nominalSpot":nominalSpot,
                    "rotation":rotation}
        with open(reference_path, 'wb') as f:
            pickle.dump(references,f)
        return references
        
    def _process_all_images(self, folder_path: Path, 
                           use_plots: bool) -> List[np.ndarray]:
        subset_list = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
        return self._process_subset_images(folder_path, subset_list, use_plots)

    def _process_subset_images(self, folder_path: Path, subset_list: List[str],
                              use_plots: bool) -> List[np.ndarray]:
        """Process subset of images."""
        surfaces = []
        
        for num, filename in enumerate(subset_list):
            if filename.endswith('.fits'):
                fileroot = filename.split('.fits')[0]
                savepath = os.path.join(folder_path, fileroot + '.npy')
                if os.path.isfile(savepath):
                    updated_surface = np.load(savepath)
                else:
                    starting_time = time.time()
                    #prepare wavefront
                    image = self._prepare_image(Path.joinpath(folder_path, filename), use_plots)
                    #compute_wavefront
                    shape_diff = self._compute_wavefront_from_image(image, use_plots)

                    updated_surface = shape_diff.copy()
                    updated_surface = updated_surface * 1e3
                    np.save(savepath, updated_surface)
                    if self._verbose:
                        print('SH image #' + str(num) + ' processed in ' + str(round(time.time() - starting_time, 1)) + ' seconds')
                surfaces.append(updated_surface)
        return surfaces
        
    def _compute_wavefront_from_image(self, image: np.ndarray, 
                                     use_plots: bool) -> np.ndarray:
        nominalSpot = self._calibration_data['nominalSpot']
        arrows, ideal_coords, actual_coords, pupil_center, pupil_radius = get_quiver(image, nominalSpot[0], nominalSpot[1], self._calibration_data["magnification"], self._pupil_definition["xyr"], use_plots)
        ptsNumX = (np.max(ideal_coords[:, 0]) - np.min(ideal_coords[:, 0])) / self._calibration_data["magnification"]
        ptsNumY = (np.max(ideal_coords[:, 1]) - np.min(ideal_coords[:, 1])) / self._calibration_data["magnification"]
        actual_spacing = self.config.lenslet_diameter_mm * self.config.mirror_focal_length_mm / self.config.reimaging_lens_focal_length_mm  # actual spacing on the primary mirror indicated by micro lenses. Units = mm
        N = round(self.config.mirror_diameter_mm / actual_spacing) # number of micro lenses on the diameter
        d = self.config.star_separation_pixels / self.config.star_separation_rad  # calibrated distance in pixels between the lens array and sensor
        slope = 0.5 * arrows / d  # calibrated slope

        #magnification is the number of pixels from one lenslet focus to the next
        #so lateral_magnification is the distance on the mirror corresponding to one pixel on the SH sensor
        lateralMagnification = actual_spacing / self._calibration_data["magnification"]  # actual spacing on the primary mirror covered by a pixel
        regularSlopeX, regularSlopeY, xCoordinates, yCoordinates = quiver2regular_slope(slope, ideal_coords, self.config.slopeMagnification, 
                                                                                        lateralMagnification, self.config.integration_step_mm, use_plots, 
                                                                                        pupil_center, pupil_radius, interpolation='RBF')
        mirrorPos = lateralMagnification * ideal_coords
        regularSlopeX, regularSlopeY = trim_maps_to_square(regularSlopeX, regularSlopeY)

        shape_diff = SouthwellIntegration(regularSlopeX,regularSlopeY)

        return shape_diff
        
    def _prepare_image(self, image_path: Path, use_plots) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare image for processing (rotation, cropping)."""
        # Implementation moved from prepare_image()
        image,_ = prepare_image(image_path, self._calibration_data["rotation"], self._pupil_definition["xyr"], output_plots=use_plots)
        return image
        
    def _analyze_surfaces(self, surfaces: List[np.ndarray], 
                         folder_path: Path) -> ReconstructionResult:
        """Analyze processed surfaces and compute metrics."""
        mean_surface = np.mean(surfaces, axis=0)
        surface_um = np.flipud(np.fliplr(mean_surface * 1e3))
        
        # Remove specified Zernike modes
        if self.config.remove_modes:
            Z = self._get_zernike_matrix(surface_um.shape[0])
            M, C = self._get_M_and_C(surface_um, Z)
            surface_um = self._remove_modes(M, C, Z, self.config.remove_modes)
            
        # Compute metrics
        rms_nm, pv_nm = self._compute_surface_metrics(surface_um)
        
        return ReconstructionResult(
            surface_um=surface_um,
            mean_surface=mean_surface, 
            rms_nm=rms_nm,
            pv_nm=pv_nm
        )
        
    def _compute_surface_metrics(self, surface: np.ndarray) -> Tuple[float, float]:
        """Compute RMS and PV metrics for surface."""
        vals = surface[~np.isnan(surface)]
        sorted_vals = np.sort(vals)
        sorted_index = int(self.config.pv_outlier_fraction * len(sorted_vals))
        
        pv = sorted_vals[-sorted_index] - sorted_vals[sorted_index]
        rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals))
        
        return rms, pv
        
    def _get_zernike_matrix(self, size: int) -> np.ndarray:
        """Get Zernike matrix for given size."""
        return General_zernike_matrix(
            self.config.zernike_modes,
            int(self.config.mirror_diameter_mm / 2 * 1e3),  # outer radius in microns
            int(self.config.mirror_inner_diameter_mm / 2 * 1e3),   # inner radius in microns
            size
        )
        
    def _get_M_and_C(self, surface: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get M and C matrices for Zernike decomposition."""
        return get_M_and_C(surface, Z)
        
    def _remove_modes(self, M: np.ndarray, C: np.ndarray, Z: np.ndarray, 
                     modes: List[int]) -> np.ndarray:
        """Remove specified Zernike modes."""
        return remove_modes(M, C, Z, modes)
        # Implementation moved from remove_modes()
        pass

#%%
# Usage example:
if __name__ == "__main__":
    """Example usage of refactored SHWFS reconstruction with output control."""

    sh_path = Path("C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/on-sky/20250626/000207/")

    # Method 1: Configure default behavior in config
    config = SHWFSConfig(
        remove_modes=[0, 1, 2, 4],
        default_output_plots=False  # Default: no plots
    )
    
    reconstructor = SHWFSReconstructor(config)
    
    # Method 2: Set instance-level output control
    reconstructor.set_output_plots(True)  # Enable plots for all operations
    
    # Calibrate once
    reconstructor.calibrate(
        sh_path=sh_path,
        folder_path=sh_path,
        redefine_pupil=False
    )
#%%    
    # Method 3: Method-level override (takes highest priority)
    result1 = reconstructor.reconstruct_from_folder(
        folder_path=sh_path,
        output_plots=True  # Force plots for this operation only
    )
#%%    
    # Method 4: Use instance setting (plots enabled from step 2)
    result2 = reconstructor.reconstruct_from_folder(
        folder_path=Path("path/to/other/images")
        # Uses self._output_plots = True
    )
    
