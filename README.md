# fish_detector_edge_ML
Endangered Fish detection using OpenVINO based edge application

Marine life constitutes half of earth’s total biodiversity. But preservation and monitoring of them is difficult due to technological limitations. Also, when fishermen and sailors catch fishes at large numbers by fishnet that include a lot of endangered species which are not edible. They are caught in the fishnet and die. To prevent catching endangered species or endangered fishes, underwater image processing, object detection, classification and implementing a sustainable solution have always been a challenge to accomplish.  
In this work, we have developed a machine learning solution that can detect and count number of fishes in a flock. This can be further trained to detect endangered fishes and species in ocean. The models can be deployed at edge devices like buoy or boats or other floating devices using Intel OpenVINO framework. The devices can monitor the count of endangered species and can show an alert message on their display. Seeing the alert message, fishermen and sailors can skip those places in sea where flock of endangered fishes is present. This can also be used to reseach on marine life.

The script can detect fishes and count from video. The script can be executed as:
python fish_detector.py <folder location of video>
