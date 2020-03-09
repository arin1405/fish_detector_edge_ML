# fish_detector_edge_ML

## Detection of Endangered Marine Fauna

Marine life constitutes half of earth’s total biodiversity. But preservation and monitoring of them is difficult due to technological limitations and reachability. Also, when fishermen and sailors catch fishes at large numbers by fishnet that include a lot of endangered species which are not edible. They are unnecessarily caught in the fishnet and die. To prevent catching endangered species or endangered fishes, underwater image processing, object detection, classification and implementing a sustainable solution have always been a challenge to accomplish.
In this work, we have developed a machine learning solution that can detect and count number of fishes in a flock. This can be further trained to detect endangered fishes and species in ocean. The models can be deployed at edge devices like buoy or boats or other floating devices using Intel OpenVINO framework. The devices can monitor the count of endangered species and can show an alert message on their display. Seeing the alert message, fishermen and sailors can skip those places in sea where flock of endangered fishes is present. This can also be used to research on marine life.

The script can detect fishes and count from video. The script can be executed as: 

> python fish_detector.py <folder location>

Where <folder location> is the location of folder containing the sample video (“fish_undetected.mp4”) and the pb file of frozen inference graph. Frames will be extracted from the video in the same folder and output video (“test.avi”) with detection and count will be created in the same folder.
