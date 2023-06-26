<!-- markdownlint-disable -->

<a href="https://github.com/MrYassinox/aruco_toolkit/blob/main\aruco_toolkit\modules\toolkit.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `toolkit.py`




**Global Variables**
---------------
- **TYPE_ARUCO_DICT**
- **DICT_4X4_50**
- **DICT_4X4_100**
- **DICT_4X4_250**
- **DICT_4X4_1000**
- **DICT_5X5_50**
- **DICT_5X5_100**
- **DICT_5X5_250**
- **DICT_5X5_1000**
- **DICT_6X6_50**
- **DICT_6X6_100**
- **DICT_6X6_250**
- **DICT_6X6_1000**
- **DICT_7X7_50**
- **DICT_7X7_100**
- **DICT_7X7_250**
- **DICT_7X7_1000**
- **DICT_APRILTAG_16h5**
- **DICT_APRILTAG_25h9**
- **DICT_APRILTAG_36h10**
- **DICT_APRILTAG_36h11**
- **DICT_ARUCO_ORIGINAL**


---

## <kbd>class</kbd> `ArUcoMarkers`
Class for generating and detecting ArUco markers using OpenCV 

<a href="https://github.com/MrYassinox/aruco_toolkit/blob/main\aruco_toolkit\modules\toolkit.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(logs: bool = True)
```

Initialize the ArUcoMarkers object. 



**Args:**
 
 - <b>``logs` (bool, optional)`</b>:  Determines whether to enable logging. Defaults to True. 




---

<a href="https://github.com/MrYassinox/aruco_toolkit/blob/main\aruco_toolkit\modules\toolkit.py#L698"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `detected_aruco_type`

```python
detected_aruco_type(image: str)
```

Detects the type of ArUco markers present in the given image. 



**Args:**
 
 - <b>``image` (str)`</b>:  The file path of the image. 



**Returns:**
 
 - <b>`str`</b>:  The name of the detected ArUco dictionary type. 



**Raises:**
 
 - <b>`Exception`</b>:  If any error occurs during the marker detection process. 



**Example:**
 ```python
    scanner = ArucoMarkerScanner()
    aruco_type = scanner.detected_aruco_type(image="image.jpg")
    print(f"Detected ArUco type: {aruco_type}")
``` 



**Notes:**

> - The `image` parameter should be a file path pointing to an image file containing ArUco markers. - The method returns the name of the detected ArUco dictionary type. - If multiple ArUco dictionaries are detected in the image, the method returns the type of the first dictionary found. - The detected markers and their IDs will be logged in the console if the `logs` attribute of the class instance is set to True. 

---

<a href="https://github.com/MrYassinox/aruco_toolkit/blob/main\aruco_toolkit\modules\toolkit.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `detected_markers`

```python
detected_markers(
    image: str,
    type_aruco: Optional[ForwardRef('DICT_4X4_50'), str] = 'DICT_4X4_50',
    resize: int = None
)
```

Detects ArUco markers in an image. 



**Args:**
 
 - <b>``image` (str)`</b>:  The path to the image file to detect markers in. 
 - <b>``type_aruco` (Union[int, str], optional)`</b>:  The type of ArUco marker to detect. It can be a predefined dictionary or a string identifier. Defaults to cv2.aruco.DICT_4X4_50. 
 - <b>``resize` (int, optional)`</b>:  The size to resize the image before detection. If not provided, the original size is used. 



**Returns:**
 
 - <b>`Tuple`</b>:  A tuple containing the detected marker corners and IDs. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the ArUco marker dictionary type is invalid. 
 - <b>`FileNotFoundError`</b>:  If the image file is not found. 



**Example:**
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

---

<a href="https://github.com/MrYassinox/aruco_toolkit/blob/main\aruco_toolkit\modules\toolkit.py#L385"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `detected_markers_camera`

```python
detected_markers_camera(
    type_aruco: Optional[ForwardRef('DICT_4X4_50'), str] = 'DICT_4X4_50',
    cameraID: int = 0,
    resize: int = None
)
```

Detects ArUco markers in real-time using a camera stream. 



**Args:**
 
 - <b>``type_aruco` (Optional[Union[DICT_4X4_50, str]])`</b>:  The type of ArUco dictionary to use for marker detection.  Defaults to DICT_4X4_50. 
 - <b>``cameraID` (int)`</b>:  The ID of the camera device to use for capturing frames. Defaults to 0. 
 - <b>``resize` (int)`</b>:  The width to resize the frames, preserving the aspect ratio. Defaults to None. 



**Returns:**
 None 



**Raises:**
 
 - <b>`Exception`</b>:  If any error occurs during the marker detection process. 



**Example:**
 ```python
    detector = ArucoMarkerDetector()
    detector.detected_markers_camera(type_aruco=DICT_4X4_50, cameraID=0, resize=800)
``` 



**Notes:**

> - The `type_aruco` parameter can be either a predefined ArUco dictionary from OpenCV or a custom dictionary filename. Supported predefined dictionaries are: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000. - The `cameraID` parameter specifies the ID of the camera device to use. It defaults to 0, which represents the default system camera. - The `resize` parameter allows resizing the frames to a specified width while maintaining the aspect ratio. - Press the 'q' key to stop the camera stream and close the window. 

---

<a href="https://github.com/MrYassinox/aruco_toolkit/blob/main\aruco_toolkit\modules\toolkit.py#L279"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `detected_markers_image`

```python
detected_markers_image(
    image: str,
    type_aruco: Optional[ForwardRef('DICT_4X4_50'), str] = 'DICT_4X4_50',
    resize: int = None,
    show: bool = False
)
```

Detects ArUco markers in an image. 



**Args:**
 
 - <b>``image` (str)`</b>:  The filename of the image to be processed. 
 - <b>``type_aruco` (Optional[Union[DICT_4X4_50, str]])`</b>:  The type of ArUco dictionary to use for marker detection.  Defaults to DICT_4X4_50. 
 - <b>``resize` (int)`</b>:  The width to resize the image, preserving the aspect ratio. Defaults to None. 
 - <b>``show` (bool)`</b>:  Whether to display the processed image with detected markers. Defaults to False. 



**Returns:**
 None 



**Raises:**
 
 - <b>`Exception`</b>:  If any error occurs during the marker detection process. 



**Example:**
 ```python
    detector = ArucoMarkerDetector()
    detector.detected_markers_image(image="markers.jpg", type_aruco=DICT_4X4_50, resize=800, show=True)
``` 



**Notes:**

> - The `image` parameter should specify the filename of the image in a supported format (e.g., JPEG, PNG). - The `type_aruco` parameter can be either a predefined ArUco dictionary from OpenCV or a custom dictionary filename. Supported predefined dictionaries are: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000. - The `resize` parameter allows resizing the image to a specified width while maintaining the aspect ratio. - Setting `show` to True will display the processed image with detected markers. Press 'q' to close the image window. 

---

<a href="https://github.com/MrYassinox/aruco_toolkit/blob/main\aruco_toolkit\modules\toolkit.py#L502"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `detected_markers_scanner`

```python
detected_markers_scanner(
    type_aruco: Optional[ForwardRef('DICT_4X4_50'), str] = 'DICT_4X4_50',
    cameraID: int = 0,
    resize: int = None
)
```

Scans ArUco markers using a camera stream and returns the detected marker IDs. 



**Args:**
 
 - <b>``type_aruco` (Optional[Union[DICT_4X4_50, str]])`</b>:  The type of ArUco dictionary to use for marker detection.  Defaults to DICT_4X4_50. 
 - <b>``cameraID` (int)`</b>:  The ID of the camera device to use for capturing frames. Defaults to 0. 
 - <b>``resize` (int)`</b>:  The width to resize the frames, preserving the aspect ratio. Defaults to None. 



**Returns:**
 
 - <b>`List[int]`</b>:  A list of detected marker IDs. 



**Raises:**
 
 - <b>`Exception`</b>:  If any error occurs during the marker detection process. 



**Example:**
 ```python
    scanner = ArucoMarkerScanner()
    marker_ids = scanner.detected_markers_scanner(type_aruco=DICT_4X4_50, cameraID=0, resize=800)
    print(marker_ids)
``` 



**Notes:**

> - The `type_aruco` parameter can be either a predefined ArUco dictionary from OpenCV or a custom dictionary filename. Supported predefined dictionaries are: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000. - The `cameraID` parameter specifies the ID of the camera device to use. It defaults to 0, which represents the default system camera. - The `resize` parameter allows resizing the frames to a specified width while maintaining the aspect ratio. - Press the 'q' key to stop the camera stream and close the window. 

---

<a href="https://github.com/MrYassinox/aruco_toolkit/blob/main\aruco_toolkit\modules\toolkit.py#L601"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `detected_markers_scanner_live`

```python
detected_markers_scanner_live(
    type_aruco: Optional[ForwardRef('DICT_4X4_50'), str] = 'DICT_4X4_50',
    cameraID: int = 0,
    resize: int = None
)
```

Continuously scans ArUco markers using a live camera stream. 



**Args:**
 
 - <b>``type_aruco` (Optional[Union[DICT_4X4_50, str]])`</b>:  The type of ArUco dictionary to use for marker detection.  Defaults to DICT_4X4_50. 
 - <b>``cameraID` (int)`</b>:  The ID of the camera device to use for capturing frames. Defaults to 0. 
 - <b>``resize` (int)`</b>:  The width to resize the frames, preserving the aspect ratio. Defaults to None. 



**Returns:**
 None 



**Raises:**
 
 - <b>`Exception`</b>:  If any error occurs during the marker detection process. 



**Example:**
 ```python
    scanner = ArucoMarkerScanner()
    scanner.detected_markers_scanner_live(type_aruco=DICT_4X4_50, cameraID=0, resize=800)
``` 



**Notes:**

> - The `type_aruco` parameter can be either a predefined ArUco dictionary from OpenCV or a custom dictionary filename. Supported predefined dictionaries are: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000. - The `cameraID` parameter specifies the ID of the camera device to use. It defaults to 0, which represents the default system camera. - The `resize` parameter allows resizing the frames to a specified width while maintaining the aspect ratio. - The detected marker IDs will be logged in real-time. - Press the 'q' key to stop the camera stream and close the window. 

---

<a href="https://github.com/MrYassinox/aruco_toolkit/blob/main\aruco_toolkit\modules\toolkit.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `generate_markers`

```python
generate_markers(
    type_aruco: Optional[ForwardRef('DICT_4X4_50'), str] = 'DICT_4X4_50',
    marker_aruco_id: int = 1,
    size_marker: int = 100,
    output_path: Optional[str] = None,
    show: bool = False
)
```

Generate ArUco markers. 



**Args:**
 
 - <b>``type_aruco` (Union[DICT_4X4_50, str], optional)`</b>:  The type of ArUco marker to generate. Defaults to DICT_4X4_50. 
 - <b>``marker_aruco_id` (int, optional)`</b>:  The ID of the ArUco marker to generate. Defaults to 1. 
 - <b>``size_marker` (int, optional)`</b>:  The size of the ArUco marker in pixels. Defaults to 100. 
 - <b>``output_path` (str, optional)`</b>:  The output path to save the generated marker. Defaults to None. 
 - <b>``show` (bool, optional)`</b>:  Determines whether to display the generated marker. Defaults to False. 



**Raises:**
 
 - <b>`Exception`</b>:  If an error occurs during marker generation. 



**Example:**
 ```python
    from aruco_toolkit.modules.toolkit import DICT_4X4_50
    aruco = ArUcoMarkers()
    aruco.generate_markers(type_aruco=DICT_4X4_50, marker_aruco_id=1, size_marker=100, output_path="markers")
``` 


