"""
DICOM Processing Core Module
Handles medical imaging DICOM files for chest X-ray analysis
"""

import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DicomMetadata:
    """Structured DICOM metadata for medical analysis"""
    patient_id: str
    patient_age: Optional[str]
    patient_sex: Optional[str]
    study_date: Optional[str]
    study_time: Optional[str]
    study_description: Optional[str]
    series_description: Optional[str]
    modality: str
    manufacturer: Optional[str]
    manufacturer_model: Optional[str]
    institution_name: Optional[str]
    station_name: Optional[str]
    image_dimensions: Tuple[int, int]
    pixel_spacing: Optional[Tuple[float, float]]
    bits_allocated: int
    bits_stored: int
    photometric_interpretation: str
    window_center: Optional[float]
    window_width: Optional[float]
    kvp: Optional[float]
    exposure_time: Optional[float]
    tube_current: Optional[float]
    view_position: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'patient_id': self.patient_id,
            'patient_age': self.patient_age,
            'patient_sex': self.patient_sex,
            'study_date': self.study_date,
            'study_time': self.study_time,
            'study_description': self.study_description,
            'series_description': self.series_description,
            'modality': self.modality,
            'manufacturer': self.manufacturer,
            'manufacturer_model': self.manufacturer_model,
            'institution_name': self.institution_name,
            'station_name': self.station_name,
            'image_dimensions': self.image_dimensions,
            'pixel_spacing': self.pixel_spacing,
            'bits_allocated': self.bits_allocated,
            'bits_stored': self.bits_stored,
            'photometric_interpretation': self.photometric_interpretation,
            'window_center': self.window_center,
            'window_width': self.window_width,
            'kvp': self.kvp,
            'exposure_time': self.exposure_time,
            'tube_current': self.tube_current,
            'view_position': self.view_position
        }

class DicomProcessor:
    """Professional DICOM processing for medical imaging"""
    
    SUPPORTED_MODALITIES = {'CR', 'DX', 'MG'}  # Chest X-ray modalities
    SUPPORTED_PHOTOMETRIC = {'MONOCHROME1', 'MONOCHROME2'}
    
    @staticmethod
    def validate_dicom_file(dicom_data: bytes) -> bool:
        """Validate if file is a proper DICOM"""
        try:
            pydicom.dcmread(io.BytesIO(dicom_data))
            return True
        except InvalidDicomError:
            return False
        except Exception as e:
            logger.warning(f"DICOM validation error: {e}")
            return False
    
    @staticmethod
    def extract_metadata(dataset: pydicom.Dataset) -> DicomMetadata:
        """Extract comprehensive medical metadata from DICOM"""
        
        def safe_get(attr: str, default=None):
            """Safely extract DICOM attribute"""
            try:
                return getattr(dataset, attr, default)
            except Exception:
                return default
        
        # Patient information (anonymized for demo)
        patient_id = safe_get('PatientID', 'ANONYMOUS')
        if patient_id and len(patient_id) > 8:
            patient_id = f"PAT_{patient_id[:4]}***"  # Anonymize
            
        # Image dimensions
        rows = safe_get('Rows', 0)
        cols = safe_get('Columns', 0)
        
        # Pixel spacing (mm/pixel)
        pixel_spacing = safe_get('PixelSpacing')
        if pixel_spacing:
            pixel_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))
        
        # Window/Level settings for display
        window_center = safe_get('WindowCenter')
        window_width = safe_get('WindowWidth')
        
        if isinstance(window_center, (list, tuple)):
            window_center = float(window_center[0]) if window_center else None
        elif window_center is not None:
            window_center = float(window_center)
            
        if isinstance(window_width, (list, tuple)):
            window_width = float(window_width[0]) if window_width else None
        elif window_width is not None:
            window_width = float(window_width)
        
        return DicomMetadata(
            patient_id=patient_id,
            patient_age=safe_get('PatientAge'),
            patient_sex=safe_get('PatientSex'),
            study_date=safe_get('StudyDate'),
            study_time=safe_get('StudyTime'),
            study_description=safe_get('StudyDescription'),
            series_description=safe_get('SeriesDescription'),
            modality=safe_get('Modality', 'UNKNOWN'),
            manufacturer=safe_get('Manufacturer'),
            manufacturer_model=safe_get('ManufacturerModelName'),
            institution_name=safe_get('InstitutionName'),
            station_name=safe_get('StationName'),
            image_dimensions=(cols, rows),
            pixel_spacing=pixel_spacing,
            bits_allocated=safe_get('BitsAllocated', 16),
            bits_stored=safe_get('BitsStored', 16),
            photometric_interpretation=safe_get('PhotometricInterpretation', 'MONOCHROME2'),
            window_center=window_center,
            window_width=window_width,
            kvp=safe_get('KVP'),
            exposure_time=safe_get('ExposureTime'),
            tube_current=safe_get('XRayTubeCurrent'),
            view_position=safe_get('ViewPosition')
        )
    
    @staticmethod
    def apply_window_level(pixel_array: np.ndarray, 
                          window_center: Optional[float], 
                          window_width: Optional[float]) -> np.ndarray:
        """Apply medical windowing/leveling to optimize visualization"""
        if window_center is None or window_width is None:
            # Auto-calculate window/level from image statistics
            window_center = float(np.mean(pixel_array))
            window_width = float(np.std(pixel_array) * 4)
        
        # Apply windowing formula
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        
        windowed = np.clip(pixel_array, img_min, img_max)
        windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        return windowed
    
    @staticmethod
    def process_pixel_data(dataset: pydicom.Dataset, 
                          metadata: DicomMetadata) -> np.ndarray:
        """Extract and process pixel data from DICOM with proper medical handling"""
        
        if not hasattr(dataset, 'pixel_array'):
            raise ValueError("DICOM file contains no pixel data")
        
        pixel_array = dataset.pixel_array.astype(np.float32)
        
        # Handle photometric interpretation
        if metadata.photometric_interpretation == 'MONOCHROME1':
            # MONOCHROME1: 0 = white, invert for standard display
            max_val = 2**metadata.bits_stored - 1
            pixel_array = max_val - pixel_array
            logger.info("Applied MONOCHROME1 inversion")
        
        # Apply medical windowing for optimal visualization
        windowed_array = DicomProcessor.apply_window_level(
            pixel_array, 
            metadata.window_center, 
            metadata.window_width
        )
        
        return windowed_array
    
    @classmethod
    def load_dicom(cls, dicom_data: bytes) -> Tuple[Image.Image, DicomMetadata]:
        """Load DICOM file and return PIL Image + metadata"""
        
        try:
            # Parse DICOM
            dataset = pydicom.dcmread(io.BytesIO(dicom_data))
            
            # Extract metadata
            metadata = cls.extract_metadata(dataset)
            
            # Validate modality
            if metadata.modality not in cls.SUPPORTED_MODALITIES:
                logger.warning(f"Unsupported modality: {metadata.modality}. Processing anyway.")
            
            # Validate photometric interpretation  
            if metadata.photometric_interpretation not in cls.SUPPORTED_PHOTOMETRIC:
                raise ValueError(f"Unsupported photometric interpretation: {metadata.photometric_interpretation}")
            
            # Process pixel data
            pixel_array = cls.process_pixel_data(dataset, metadata)
            
            # Convert to PIL Image
            image = Image.fromarray(pixel_array, mode='L')
            
            logger.info(f"Successfully loaded DICOM: {metadata.modality} "
                       f"{metadata.image_dimensions[0]}x{metadata.image_dimensions[1]} "
                       f"{metadata.bits_allocated}-bit")
            
            return image, metadata
            
        except InvalidDicomError:
            raise ValueError("Invalid DICOM file format")
        except Exception as e:
            raise ValueError(f"Error processing DICOM: {str(e)}")
    
    @staticmethod
    def is_chest_xray(metadata: DicomMetadata) -> bool:
        """Determine if DICOM is likely a chest X-ray"""
        chest_keywords = ['chest', 'thorax', 'lung', 'pulmonary', 'cardiac']
        
        # Check study/series descriptions
        study_desc = (metadata.study_description or '').lower()
        series_desc = (metadata.series_description or '').lower()
        
        for keyword in chest_keywords:
            if keyword in study_desc or keyword in series_desc:
                return True
        
        # Check modality (CR/DX are common for chest X-rays)
        if metadata.modality in ['CR', 'DX']:
            return True
            
        return False
    
    @staticmethod
    def anonymize_metadata(metadata: DicomMetadata) -> DicomMetadata:
        """Remove/mask personally identifiable information"""
        anonymized = metadata
        
        # Mask patient ID
        if metadata.patient_id and metadata.patient_id != 'ANONYMOUS':
            anonymized.patient_id = f"PAT_{metadata.patient_id[:3]}***"
        
        # Remove institution details for privacy
        anonymized.institution_name = None
        anonymized.station_name = None
        
        return anonymized

# Utility functions for common operations
def validate_chest_xray_dicom(dicom_data: bytes) -> Tuple[bool, str]:
    """Validate if DICOM is suitable for chest X-ray analysis"""
    try:
        image, metadata = DicomProcessor.load_dicom(dicom_data)
        
        if not DicomProcessor.is_chest_xray(metadata):
            return False, f"Not identified as chest X-ray (modality: {metadata.modality})"
        
        # Check minimum image dimensions
        width, height = metadata.image_dimensions
        if width < 224 or height < 224:
            return False, f"Image too small for analysis: {width}x{height}"
        
        return True, "Valid chest X-ray DICOM"
        
    except Exception as e:
        return False, f"DICOM validation failed: {str(e)}"