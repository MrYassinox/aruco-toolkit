''' generating and detecting ArUco markers using OpenCV. '''
########################################################################################################################
# TODO DOCUMENT
########################################################################################################################
# |   ├──  | 
# |   |    └── doc: 

########################################################################################################################
# TODO IMPORTING NECESSARY LIBRARYIES
########################################################################################################################
# LIB => python libraryies
import os
import sys
import time
from typing import List, Any, Dict, Callable, Union, Tuple, Iterable, Literal, Optional

# LIB => squardot-utils-standard libraryies
from utils_standard.modules.utils import LoggerHandle

# LIB => opencv libraryies
import cv2

# LIB => imutils libraryies
import imutils

# LIB => numpy libraryies
import numpy as np

########################################################################################################################
# TODO SET UP
########################################################################################################################
# NOTE => define names of each possible ArUco tag OpenCV supports
TYPE_ARUCO_DICT = {
    # DESC => The cv2.aruco.DICT_4X4_50 value implies that we want to generate a binary 4×4 square AruCo marker. We’ll be able to generate 50 unique ArUco marker IDs using this dictionary.
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50, # DESC => The cv2.aruco.DICT_4X4_50 value implies that we want to generate a binary 4×4 square AruCo marker. We’ll be able to generate 50 unique ArUco marker IDs using this dictionary.
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "DICT_APRILTAG_16H5": cv2.aruco.DICT_APRILTAG_16H5,
    "DICT_APRILTAG_25H9": cv2.aruco.DICT_APRILTAG_25H9,
    "DICT_APRILTAG_36H10": cv2.aruco.DICT_APRILTAG_36H10,
    "DICT_APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36H11
}

# NOTE => define names of each possible ArUco tag
DICT_4X4_50 = 'DICT_4X4_50'
DICT_4X4_100 = 'DICT_4X4_100'
DICT_4X4_250 = 'DICT_4X4_250'
DICT_4X4_1000 = 'DICT_4X4_1000'
DICT_5X5_50 = 'DICT_5X5_50'
DICT_5X5_100 = 'DICT_5X5_100'
DICT_5X5_250 = 'DICT_5X5_250'
DICT_5X5_1000 = 'DICT_5X5_1000'
DICT_6X6_50 = 'DICT_6X6_50'
DICT_6X6_100 = 'DICT_6X6_100'
DICT_6X6_250 = 'DICT_6X6_250'
DICT_6X6_1000 = 'DICT_6X6_1000'
DICT_7X7_50 = 'DICT_7X7_50'
DICT_7X7_100 = 'DICT_7X7_100'
DICT_7X7_250 = 'DICT_7X7_250'
DICT_7X7_1000 = 'DICT_7X7_1000'
DICT_APRILTAG_16h5 = 'DICT_APRILTAG_16h5'
DICT_APRILTAG_25h9 = 'DICT_APRILTAG_25h9'
DICT_APRILTAG_36h10 = 'DICT_APRILTAG_36h10'
DICT_APRILTAG_36h11 = 'DICT_APRILTAG_36h11'
DICT_ARUCO_ORIGINAL = 'DICT_ARUCO_ORIGINAL'

logger = LoggerHandle(context='ArUco')

########################################################################################################################
# TODO FUNCTIONS MODULES
########################################################################################################################

########################################################################################################################
# TODO CLASSES MODULES
########################################################################################################################
class ArUcoMarkers:
    """Class for generating and detecting ArUco markers using OpenCV"""
    def __init__(self, logs: bool=True): # DESC => initialize constructor
        """Initialize the ArUcoMarkers object.

        Args:
            `logs` (bool, optional): Determines whether to enable logging. Defaults to True.
        """
        super(ArUcoMarkers, self).__init__()
        # DESC => store the value attribute
        self.logs = logs
		
        # DESC => define names of each possible ArUco tag OpenCV supports
        self.TYPE_ARUCO_DICT = {
			# DESC => The cv2.aruco.DICT_4X4_50 value implies that we want to generate a binary 4×4 square AruCo marker. We’ll be able to generate 50 unique ArUco marker IDs using this dictionary.
			"DICT_4X4_50": cv2.aruco.DICT_4X4_50, # DESC => The cv2.aruco.DICT_4X4_50 value implies that we want to generate a binary 4×4 square AruCo marker. We’ll be able to generate 50 unique ArUco marker IDs using this dictionary.
			"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
			"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
			"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
			"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
			"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
			"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
			"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
			"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
			"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
			"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
			"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
			"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
			"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
			"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
			"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
			"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
			"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
			"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
			"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
			"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
			"DICT_APRILTAG_16H5": cv2.aruco.DICT_APRILTAG_16H5,
			"DICT_APRILTAG_25H9": cv2.aruco.DICT_APRILTAG_25H9,
			"DICT_APRILTAG_36H10": cv2.aruco.DICT_APRILTAG_36H10,
			"DICT_APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36H11
		}

		# DESC => define names of each possible ArUco tag
        self.DICT_4X4_50 = 'DICT_4X4_50'
        self.DICT_4X4_100 = 'DICT_4X4_100'
        self.DICT_4X4_250 = 'DICT_4X4_250'
        self.DICT_4X4_1000 = 'DICT_4X4_1000'
        self.DICT_5X5_50 = 'DICT_5X5_50'
        self.DICT_5X5_100 = 'DICT_5X5_100'
        self.DICT_5X5_250 = 'DICT_5X5_250'
        self.DICT_5X5_1000 = 'DICT_5X5_1000'
        self.DICT_6X6_50 = 'DICT_6X6_50'
        self.DICT_6X6_100 = 'DICT_6X6_100'
        self.DICT_6X6_250 = 'DICT_6X6_250'
        self.DICT_6X6_1000 = 'DICT_6X6_1000'
        self.DICT_7X7_50 = 'DICT_7X7_50'
        self.DICT_7X7_100 = 'DICT_7X7_100'
        self.DICT_7X7_250 = 'DICT_7X7_250'
        self.DICT_7X7_1000 = 'DICT_7X7_1000'
        self.DICT_APRILTAG_16h5 = 'DICT_APRILTAG_16h5'
        self.DICT_APRILTAG_25h9 = 'DICT_APRILTAG_25h9'
        self.DICT_APRILTAG_36h10 = 'DICT_APRILTAG_36h10'
        self.DICT_APRILTAG_36h11 = 'DICT_APRILTAG_36h11'
        self.DICT_ARUCO_ORIGINAL = 'DICT_ARUCO_ORIGINAL'

    def generate_markers(self, type_aruco: Optional[Union[DICT_4X4_50, str]]=DICT_4X4_50, marker_aruco_id: int=1, size_marker: int=100, output_path: Optional[str]=None, show: bool=False):
        """Generate ArUco markers.

        Args:
            `type_aruco` (Union[DICT_4X4_50, str], optional): The type of ArUco marker to generate. Defaults to DICT_4X4_50.
            `marker_aruco_id` (int, optional): The ID of the ArUco marker to generate. Defaults to 1.
            `size_marker` (int, optional): The size of the ArUco marker in pixels. Defaults to 100.
            `output_path` (str, optional): The output path to save the generated marker. Defaults to None.
            `show` (bool, optional): Determines whether to display the generated marker. Defaults to False.

        Raises:
            Exception: If an error occurs during marker generation.

        Example:
        ```python
            from aruco_toolkit.modules.toolkit import DICT_4X4_50
            aruco = ArUcoMarkers()
            aruco.generate_markers(type_aruco=DICT_4X4_50, marker_aruco_id=1, size_marker=100, output_path="markers")
        ```
        """
        try:
            # DESC => verify that the supplied ArUCo tag exists and is supported by OpenCV
            if self.TYPE_ARUCO_DICT.get(type_aruco, None) is None:
                logger.info(f"ArUCo tag of '{type_aruco}' is not supported") if self.logs else None
            else:
                logger.info(f"Generating ArUCo tag type '{type_aruco}' with ID '{marker_aruco_id}'") if self.logs else None

                # DESC => load the ArUCo dictionary
                get_aruco_type = cv2.aruco.getPredefinedDictionary(self.TYPE_ARUCO_DICT[type_aruco])

                # DESC => allocates memory for a 300x300x1 grayscale image. We use grayscale here, since an ArUco tag is a binary image
                img_empty = np.zeros(shape=(size_marker, size_marker, 1), dtype=np.uint8)
                gen_marker = cv2.aruco.generateImageMarker(dictionary=get_aruco_type, id=marker_aruco_id, sidePixels=size_marker, img=img_empty, borderBits=1)

                if output_path is not None:
                    output_filename = os.path.join(output_path, '{}_ID{}.png').format(type_aruco, marker_aruco_id)
                else:
                    output_filename = os.path.join('{}_ID{}.png').format(type_aruco, marker_aruco_id)
                
                # DESC => write the generated ArUCo tag to disk
                cv2.imwrite(filename=output_filename, img=gen_marker)
                logger.info(f"Savde ArUCo tag in: {output_filename}") if self.logs else None

                # DESC => display it to our screen
                if show:
                    cv2.imshow(winname="ArUCo Generate Tag", mat=gen_marker)
                    if cv2.waitKey(0) == ord('q'):
                        cv2.destroyAllWindows()
        except Exception as error:
            raise Exception(f'Error: {error}')

    def detected_markers(self, image: str, type_aruco: Optional[Union[DICT_4X4_50, str]]=DICT_4X4_50, resize: int=None):
        """Detects ArUco markers in an image.

        Args:
            `image` (str): The path to the image file to detect markers in.
            `type_aruco` (Union[int, str], optional): The type of ArUco marker to detect. It can be a predefined dictionary or a string identifier. Defaults to cv2.aruco.DICT_4X4_50.
            `resize` (int, optional): The size to resize the image before detection. If not provided, the original size is used.

        Returns:
            Tuple: A tuple containing the detected marker corners and IDs.

        Raises:
            ValueError: If the ArUco marker dictionary type is invalid.
            FileNotFoundError: If the image file is not found.

        Example:
        ```python
            aruco = ArUcoMarkers()
            corners, ids = aruco.detected_markers(image="marker.png", type_aruco=cv2.aruco.DICT_4X4_50, resize=800)
            print(corners)

            # output.
            #
            # [[[-0.38492805  0.33673733]
            # [ 0.13139215  0.38344035]
            # [ 0.14946598 -0.19659886]
            # [-0.36935449 -0.20254551]]]
            
            print(ids)
            # output.
            #
            # [[1]]
        ```
        """
        try:
            # DESC => load and read image with opencv
            src = cv2.imread(filename=image)

            # DESC => resize the reference and input images
            if resize is not None:
                src = imutils.resize(image=src, width=resize)

            # DESC => converts color space conversion code
            img_gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)

            # DESC => verify that the supplied ArUCo tag exists and is supported by OpenCV
            if self.TYPE_ARUCO_DICT.get(type_aruco, None) is None:
                logger.info(f"ArUCo tag of '{type_aruco}' is not supported") if self.logs else None
            else:
                logger.info(f"detecting '{type_aruco}' tags...") if self.logs else None
                
                get_aruco_type = cv2.aruco.getPredefinedDictionary(self.TYPE_ARUCO_DICT[type_aruco]) # DESC => Load the ArUco dictionary
                detector_parameters = cv2.aruco.DetectorParameters() # DESC => Instantiate our ArUco detector parameters
                
                # DESC => Apply ArUco detection, and return in a 3-tuple of: 
                # DESC => corners: The (x, y)-coordinates of our detected ArUco markers, 
                # DESC => ids: The identifiers of the ArUco markers (i.e., the ID encoded in the marker itself), 
                # DESC => rejected: A list of potential markers that were detected but ultimately rejected due to the code inside the marker not being able to be parsed
                corners, ids, rejected = cv2.aruco.detectMarkers(img_gray, get_aruco_type, parameters=detector_parameters)

                # DESC => verify *at least* one ArUco marker was detected
                if len(corners) > 0:
                    # DESC => flatten the ArUco IDs list
                    marker_ids = ids.flatten()
                    logger.info(f"ArUco detected marker ID: [{marker_ids}]") if self.logs else None
                    return corners, marker_ids
        except Exception as error:
            raise Exception(f'Error: {error}')

    def detected_markers_image(self, image: str, type_aruco: Optional[Union[DICT_4X4_50, str]]=DICT_4X4_50, resize: int=None, show: bool=False):
        """Detects ArUco markers in an image.

        Args:
            `image` (str): The filename of the image to be processed.
            `type_aruco` (Optional[Union[DICT_4X4_50, str]]): The type of ArUco dictionary to use for marker detection.
                Defaults to DICT_4X4_50.
            `resize` (int): The width to resize the image, preserving the aspect ratio. Defaults to None.
            `show` (bool): Whether to display the processed image with detected markers. Defaults to False.

        Returns:
            None

        Raises:
            Exception: If any error occurs during the marker detection process.

        Example:
        ```python
            detector = ArucoMarkerDetector()
            detector.detected_markers_image(image="markers.jpg", type_aruco=DICT_4X4_50, resize=800, show=True)
        ```

        Notes:
            - The `image` parameter should specify the filename of the image in a supported format (e.g., JPEG, PNG).
            - The `type_aruco` parameter can be either a predefined ArUco dictionary from OpenCV or a custom dictionary filename.
                Supported predefined dictionaries are: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50,
                DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000,
                DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000.
            - The `resize` parameter allows resizing the image to a specified width while maintaining the aspect ratio.
            - Setting `show` to True will display the processed image with detected markers. Press 'q' to close the image window.
        """
        try:
            # DESC => load and read image with opencv
            src = cv2.imread(filename=image)
            
            # DESC => resize the reference and input images
            if resize is not None:
                src = imutils.resize(image=src, width=resize)

            # DESC => converts color space conversion code
            img_gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)

            # DESC => verify that the supplied ArUCo tag exists and is supported by OpenCV
            if self.TYPE_ARUCO_DICT.get(type_aruco, None) is None:
                logger.info(f"ArUCo tag of '{type_aruco}' is not supported") if self.logs else None
            else:
                logger.info(f"detecting '{type_aruco}' tags...") if self.logs else None
                
                get_aruco_type = cv2.aruco.getPredefinedDictionary(self.TYPE_ARUCO_DICT[type_aruco]) # DESC => Load the ArUco dictionary
                detector_parameters = cv2.aruco.DetectorParameters() # DESC => Instantiate our ArUco detector parameters
                
                # DESC => Apply ArUco detection, and return in a 3-tuple of: 
                # DESC => corners: The (x, y)-coordinates of our detected ArUco markers, 
                # DESC => ids: The identifiers of the ArUco markers (i.e., the ID encoded in the marker itself), 
                # DESC => rejected: A list of potential markers that were detected but ultimately rejected due to the code inside the marker not being able to be parsed
                corners, ids, rejected = cv2.aruco.detectMarkers(img_gray, get_aruco_type, parameters=detector_parameters)

                # DESC => verify *at least* one ArUco marker was detected
                if len(corners) > 0:
                    # DESC => flatten the ArUco IDs list
                    ids = ids.flatten()

                    # DESC => loop over the detected ArUCo corners
                    for (markerCorner, markerID) in zip(corners, ids):
                        # DESC => extract the marker corners (which are always returned in
                        # DESC => top-left, top-right, bottom-right, and bottom-left order)
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        
                        # DESC => convert each of the (x, y)-coordinate pairs to integers
                        topRight = (int(topRight[0]), int(topRight[1])) # DESC => extract the top-right marker
                        bottomRight = (int(bottomRight[0]), int(bottomRight[1])) # DESC => extract the bottom-right marker
                        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1])) # DESC => extract the bottom-left marker
                        topLeft = (int(topLeft[0]), int(topLeft[1])) # DESC => extract the top-left marker


                        # DESC => draw the bounding box of the ArUCo detection
                        cv2.line(img=src, pt1=topLeft, pt2=topRight, color=(0, 255, 0), thickness=1)
                        cv2.line(img=src, pt1=topRight, pt2=bottomRight, color=(0, 255, 0), thickness=1)
                        cv2.line(img=src, pt1=bottomRight, pt2=bottomLeft, color=(0, 255, 0), thickness=1)
                        cv2.line(img=src, pt1=bottomLeft, pt2=topLeft, color=(0, 255, 0), thickness=1)
                        
                        # DESC => draw the circle of the coordinates in ArUco marker
                        cv2.circle(img=src, center=topLeft, radius=8, color=(0, 100, 255), thickness=1)
                        cv2.circle(img=src, center=topRight, radius=8, color=(0, 100, 255), thickness=1)
                        cv2.circle(img=src, center=bottomRight, radius=8, color=(0, 100, 255), thickness=1)
                        cv2.circle(img=src, center=bottomLeft, radius=8, color=(0, 100, 255), thickness=1)

                        # DESC => compute and draw the center (x, y)-coordinates of the ArUco marker
                        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                        cv2.circle(img=src, center=(cX, cY), radius=2, color=(0, 0, 255), thickness=-1)
                        
                        # DESC => draw the ArUco marker ID on the image
                        cv2.putText(img=src, text=str(markerID), org=(topLeft[0], topLeft[1] - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 255, 0), thickness=1)
                        
                        logger.info(f"ArUco detected marker ID: [{markerID}]") if self.logs else None

                    # DESC => display it to our screen
                    if show:
                        cv2.imshow(winname="Image", mat=src)
                        if cv2.waitKey(0) == ord('q'):
                            cv2.destroyAllWindows()
        except Exception as error:
            raise Exception(f'Error: {error}')

    def detected_markers_camera(self, type_aruco: Optional[Union[DICT_4X4_50, str]]=DICT_4X4_50, cameraID: int=0, resize: int=None):
        """Detects ArUco markers in real-time using a camera stream.

        Args:
            `type_aruco` (Optional[Union[DICT_4X4_50, str]]): The type of ArUco dictionary to use for marker detection.
                Defaults to DICT_4X4_50.
            `cameraID` (int): The ID of the camera device to use for capturing frames. Defaults to 0.
            `resize` (int): The width to resize the frames, preserving the aspect ratio. Defaults to None.

        Returns:
            None

        Raises:
            Exception: If any error occurs during the marker detection process.

        Example:
        ```python
            detector = ArucoMarkerDetector()
            detector.detected_markers_camera(type_aruco=DICT_4X4_50, cameraID=0, resize=800)
        ```

        Notes:
            - The `type_aruco` parameter can be either a predefined ArUco dictionary from OpenCV or a custom dictionary filename.
                Supported predefined dictionaries are: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50,
                DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000,
                DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000.
            - The `cameraID` parameter specifies the ID of the camera device to use. It defaults to 0, which represents the default system camera.
            - The `resize` parameter allows resizing the frames to a specified width while maintaining the aspect ratio.
            - Press the 'q' key to stop the camera stream and close the window.
        """
        try:
            # DESC => initialize the video stream and allow the camera sensor to warm up
            cap = cv2.VideoCapture(cameraID)

            # DESC => verify that the supplied ArUCo tag exists and is supported by OpenCV
            if self.TYPE_ARUCO_DICT.get(type_aruco, None) is None:
                logger.info(f"ArUCo tag of '{type_aruco}' is not supported") if self.logs else None
            else:
                try:
                    logger.info("Starting camera stream...") if self.logs else None
                    
                    # DESC => loop over the frames from the video stream
                    while True:
                        ret, src = cap.read()    

                        # DESC => resize the reference and input images
                        if resize is not None:
                            src = imutils.resize(image=src, width=resize)

                        # DESC => converts color space conversion code
                        frame_gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)    

                        get_aruco_type = cv2.aruco.getPredefinedDictionary(self.TYPE_ARUCO_DICT[type_aruco]) # DESC => Load the ArUco dictionary
                        detector_parameters = cv2.aruco.DetectorParameters() # DESC => Instantiate our ArUco detector parameters
                        
                        # DESC => Apply ArUco detection, and return in a 3-tuple of: 
                        # DESC => corners: The (x, y)-coordinates of our detected ArUco markers, 
                        # DESC => ids: The identifiers of the ArUco markers (i.e., the ID encoded in the marker itself), 
                        # DESC => rejected: A list of potential markers that were detected but ultimately rejected due to the code inside the marker not being able to be parsed
                        corners, ids, rejected = cv2.aruco.detectMarkers(frame_gray, get_aruco_type, parameters=detector_parameters)
                        
                        # DESC => verify *at least* one ArUco marker was detected
                        if len(corners) > 0:
                            # DESC => flatten the ArUco IDs list
                            ids = ids.flatten()        

                            # DESC => loop over the detected ArUCo corners
                            for (markerCorner, markerID) in zip(corners, ids):
                                # DESC => extract the marker corners (which are always returned in
                                # DESC => top-left, top-right, bottom-right, and bottom-left order)
                                corners = markerCorner.reshape((4, 2))
                                (topLeft, topRight, bottomRight, bottomLeft) = corners
                                
                                # DESC => convert each of the (x, y)-coordinate pairs to integers
                                topRight = (int(topRight[0]), int(topRight[1]))
                                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                                topLeft = (int(topLeft[0]), int(topLeft[1]))        
            
                                # DESC => draw the bounding box of the ArUCo detection
                                cv2.line(img=src, pt1=topLeft, pt2=topRight, color=(0, 255, 0), thickness=1)
                                cv2.line(img=src, pt1=topRight, pt2=bottomRight, color=(0, 255, 0), thickness=1)
                                cv2.line(img=src, pt1=bottomRight, pt2=bottomLeft, color=(0, 255, 0), thickness=1)
                                cv2.line(img=src, pt1=bottomLeft, pt2=topLeft, color=(0, 255, 0), thickness=1)

                                # DESC => draw the circle of the coordinates in ArUco marker
                                cv2.circle(img=src, center=topLeft, radius=8, color=(0, 100, 255), thickness=1)
                                cv2.circle(img=src, center=topRight, radius=8, color=(0, 100, 255), thickness=1)
                                cv2.circle(img=src, center=bottomRight, radius=8, color=(0, 100, 255), thickness=1)
                                cv2.circle(img=src, center=bottomLeft, radius=8, color=(0, 100, 255), thickness=1)

                                # DESC => compute and draw the center (x, y)-coordinates of the ArUco marker
                                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                                cv2.circle(img=src, center=(cX, cY), radius=2, color=(0, 0, 255), thickness=-1)
                                
                                # DESC => draw the ArUco marker ID on the image
                                cv2.putText(img=src, text=str(markerID), org=(topLeft[0], topLeft[1] - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 255, 0), thickness=1)

                                logger.info(f"ArUco detected marker ID: [{markerID}]") if self.logs else None

                        # DESC => display it to our screen
                        cv2.imshow("Frame", src)

                        # DESC => if the `q` key was pressed, break from the loop
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            cv2.destroyAllWindows()
                            break

                        time.sleep(0.1)
                    cv2.destroyAllWindows()
                except KeyboardInterrupt:
                    cap.release()
        except Exception as error:
            raise Exception(f'Error: {error}')

    def detected_markers_scanner(self, type_aruco: Optional[Union[DICT_4X4_50, str]]=DICT_4X4_50, cameraID: int=0, resize: int=None):
        """Scans ArUco markers using a camera stream and returns the detected marker IDs.

        Args:
            `type_aruco` (Optional[Union[DICT_4X4_50, str]]): The type of ArUco dictionary to use for marker detection.
                Defaults to DICT_4X4_50.
            `cameraID` (int): The ID of the camera device to use for capturing frames. Defaults to 0.
            `resize` (int): The width to resize the frames, preserving the aspect ratio. Defaults to None.

        Returns:
            List[int]: A list of detected marker IDs.

        Raises:
            Exception: If any error occurs during the marker detection process.

        Example:
        ```python
            scanner = ArucoMarkerScanner()
            marker_ids = scanner.detected_markers_scanner(type_aruco=DICT_4X4_50, cameraID=0, resize=800)
            print(marker_ids)
        ```

        Notes:
            - The `type_aruco` parameter can be either a predefined ArUco dictionary from OpenCV or a custom dictionary filename.
                Supported predefined dictionaries are: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50,
                DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000,
                DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000.
            - The `cameraID` parameter specifies the ID of the camera device to use. It defaults to 0, which represents the default system camera.
            - The `resize` parameter allows resizing the frames to a specified width while maintaining the aspect ratio.
            - Press the 'q' key to stop the camera stream and close the window.
        """
        try:
            # DESC => initialize the video stream and allow the camera sensor to warm up
            cap = cv2.VideoCapture(cameraID)

            # DESC => verify that the supplied ArUCo tag exists and is supported by OpenCV
            if self.TYPE_ARUCO_DICT.get(type_aruco, None) is None:
                logger.info(f"ArUCo tag of '{type_aruco}' is not supported") if self.logs else None
            else:
                try:
                    logger.info("Starting camera scanner stream...") if self.logs else None

                    # DESC => loop over the frames from the video stream
                    while True:
                        ret, src = cap.read()    

                        # DESC => resize the reference and input images
                        if resize is not None:
                            src = imutils.resize(image=src, width=resize)

                        # DESC => converts color space conversion code
                        frame_gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)    

                        get_aruco_type = cv2.aruco.getPredefinedDictionary(self.TYPE_ARUCO_DICT[type_aruco]) # DESC => Load the ArUco dictionary
                        detector_parameters = cv2.aruco.DetectorParameters() # DESC => Instantiate our ArUco detector parameters
                        
                        # DESC => Apply ArUco detection, and return in a 3-tuple of: 
                        # DESC => corners: The (x, y)-coordinates of our detected ArUco markers, 
                        # DESC => ids: The identifiers of the ArUco markers (i.e., the ID encoded in the marker itself), 
                        # DESC => rejected: A list of potential markers that were detected but ultimately rejected due to the code inside the marker not being able to be parsed
                        corners, ids, rejected = cv2.aruco.detectMarkers(frame_gray, get_aruco_type, parameters=detector_parameters)
                        
                        # DESC => verify *at least* one ArUco marker was detected
                        if len(corners) > 0:
                            # DESC => flatten the ArUco IDs list
                            ids = ids.flatten()        

                            list_ids = np.ravel(ids)
                            logger.info(f"ArUco detected marker ID: {list_ids}") if self.logs else None
                            return list_ids

                            # DESC => loop over the detected ArUCo corners
                            for (markerCorner, markerID) in zip(corners, ids):
                                # DESC => extract the marker corners (which are always returned in
                                # DESC => top-left, top-right, bottom-right, and bottom-left order)
                                corners = markerCorner.reshape((4, 2))
                                (topLeft, topRight, bottomRight, bottomLeft) = corners
                                
                                # DESC => convert each of the (x, y)-coordinate pairs to integers
                                topRight = (int(topRight[0]), int(topRight[1]))
                                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                                topLeft = (int(topLeft[0]), int(topLeft[1]))        

                                logger.info(f"ArUco detected marker ID: [{markerID}]") if self.logs else None

                        # DESC => if the `q` key was pressed, break from the loop
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            cv2.destroyAllWindows()
                            break

                        time.sleep(0.4)
                    cv2.destroyAllWindows()
                except KeyboardInterrupt:
                    cap.release()
        except Exception as error:
            raise Exception(f'Error: {error}')

    def detected_markers_scanner_live(self, type_aruco: Optional[Union[DICT_4X4_50, str]]=DICT_4X4_50, cameraID: int=0, resize: int=None):
        """Continuously scans ArUco markers using a live camera stream.

        Args:
            `type_aruco` (Optional[Union[DICT_4X4_50, str]]): The type of ArUco dictionary to use for marker detection.
                Defaults to DICT_4X4_50.
            `cameraID` (int): The ID of the camera device to use for capturing frames. Defaults to 0.
            `resize` (int): The width to resize the frames, preserving the aspect ratio. Defaults to None.

        Returns:
            None

        Raises:
            Exception: If any error occurs during the marker detection process.

        Example:
        ```python
            scanner = ArucoMarkerScanner()
            scanner.detected_markers_scanner_live(type_aruco=DICT_4X4_50, cameraID=0, resize=800)
        ```

        Notes:
            - The `type_aruco` parameter can be either a predefined ArUco dictionary from OpenCV or a custom dictionary filename.
                Supported predefined dictionaries are: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50,
                DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000,
                DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000.
            - The `cameraID` parameter specifies the ID of the camera device to use. It defaults to 0, which represents the default system camera.
            - The `resize` parameter allows resizing the frames to a specified width while maintaining the aspect ratio.
            - The detected marker IDs will be logged in real-time.
            - Press the 'q' key to stop the camera stream and close the window.
        """
        try:
            # DESC => initialize the video stream and allow the camera sensor to warm up
            cap = cv2.VideoCapture(cameraID)

            # DESC => verify that the supplied ArUCo tag exists and is supported by OpenCV
            if self.TYPE_ARUCO_DICT.get(type_aruco, None) is None:
                logger.info(f"ArUCo tag of '{type_aruco}' is not supported") if self.logs else None
            else:
                try:
                    logger.info("Starting camera scanner stream...") if self.logs else None

                    # DESC => loop over the frames from the video stream
                    while True:
                        ret, src = cap.read()    

                        # DESC => resize the reference and input images
                        if resize is not None:
                            src = imutils.resize(image=src, width=resize)

                        # DESC => converts color space conversion code
                        frame_gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)    

                        get_aruco_type = cv2.aruco.getPredefinedDictionary(self.TYPE_ARUCO_DICT[type_aruco]) # DESC => Load the ArUco dictionary
                        detector_parameters = cv2.aruco.DetectorParameters() # DESC => Instantiate our ArUco detector parameters
                        
                        # DESC => Apply ArUco detection, and return in a 3-tuple of: 
                        # DESC => corners: The (x, y)-coordinates of our detected ArUco markers, 
                        # DESC => ids: The identifiers of the ArUco markers (i.e., the ID encoded in the marker itself), 
                        # DESC => rejected: A list of potential markers that were detected but ultimately rejected due to the code inside the marker not being able to be parsed
                        corners, ids, rejected = cv2.aruco.detectMarkers(frame_gray, get_aruco_type, parameters=detector_parameters)
                        

                        # DESC => verify *at least* one ArUco marker was detected
                        if len(corners) > 0:
                            # DESC => flatten the ArUco IDs list
                            ids = ids.flatten()        
                            list_ids = np.ravel(ids)

                            # DESC => loop over the detected ArUCo corners
                            for (markerCorner, markerID) in zip(corners, ids):
                                # DESC => extract the marker corners (which are always returned in
                                # DESC => top-left, top-right, bottom-right, and bottom-left order)
                                corners = markerCorner.reshape((4, 2))
                                (topLeft, topRight, bottomRight, bottomLeft) = corners
                                
                                # DESC => convert each of the (x, y)-coordinate pairs to integers
                                topRight = (int(topRight[0]), int(topRight[1]))
                                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                                topLeft = (int(topLeft[0]), int(topLeft[1]))        

                                logger.info(f"ArUco detected marker ID: [{markerID}]") if self.logs else None

                        # DESC => if the `q` key was pressed, break from the loop
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            cv2.destroyAllWindows()
                            break

                        time.sleep(0.4)
                    cv2.destroyAllWindows()
                except KeyboardInterrupt:
                    cap.release()
        except Exception as error:
            raise Exception(f'Error: {error}')

    def detected_aruco_type(self, image: str):
        """Detects the type of ArUco markers present in the given image.

        Args:
            `image` (str): The file path of the image.

        Returns:
            str: The name of the detected ArUco dictionary type.

        Raises:
            Exception: If any error occurs during the marker detection process.

        Example:
        ```python
            scanner = ArucoMarkerScanner()
            aruco_type = scanner.detected_aruco_type(image="image.jpg")
            print(f"Detected ArUco type: {aruco_type}")
        ```

        Notes:
            - The `image` parameter should be a file path pointing to an image file containing ArUco markers.
            - The method returns the name of the detected ArUco dictionary type.
            - If multiple ArUco dictionaries are detected in the image, the method returns the type of the first dictionary found.
            - The detected markers and their IDs will be logged in the console if the `logs` attribute of the class instance is set to True.
        """
        try:
            # DESC => load and read image with opencv
            src = cv2.imread(filename=image)
            
            # DESC => resize the reference and input images
            src = imutils.resize(image=src, width=600)

            # DESC => converts color space conversion code
            img_gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
            
            # DESC => loop over the types of ArUco dictionaries
            for (aruco_name, aruco_type) in self.TYPE_ARUCO_DICT.items():
                # DESC => load the ArUCo dictionary, grab the ArUCo parameters, and attempt to detect the markers for the current dictionary
                get_aruco_type = cv2.aruco.getPredefinedDictionary(aruco_type) # DESC => Load the ArUco dictionary
                detector_parameters = cv2.aruco.DetectorParameters() # DESC => Instantiate our ArUco detector parameters
                
                # DESC => Apply ArUco detection, and return in a 3-tuple of: 
                # DESC => corners: The (x, y)-coordinates of our detected ArUco markers, 
                # DESC => ids: The identifiers of the ArUco markers (i.e., the ID encoded in the marker itself), 
                # DESC => rejected: A list of potential markers that were detected but ultimately rejected due to the code inside the marker not being able to be parsed
                corners, ids, rejected = cv2.aruco.detectMarkers(img_gray, get_aruco_type, parameters=detector_parameters)

                # DESC => if at least one ArUco marker was detected display the ArUco name to our terminal
                if len(corners) > 0:
                    # DESC => flatten the ArUco IDs list
                    list_ids = np.ravel(ids)
                    logger.info(f"ArUco detected {len(corners)} markers ID {list_ids} | for aruco type: '{aruco_name}'") if self.logs else None
                    return aruco_name
        except Exception as error:
            raise Exception(f'Error: {error}')

