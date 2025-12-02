# 2LEOD: 2D LiDAR Ensemble Opponent Detector
The repository for using 2D LiDAR scan data and estimates real-time object opponent odometry. We use 3 approaches which is linear regression(2 wall line fitting), logistic regression(naively using each points of scan data), and DBSCAN(rule-based). 
We check the DBSCAN is slower, fails on some scenarios, and less accurate. We use learning-based approach to overcome those limitations. 


### CSV format
- rule: one css file for one rosbag file. Must check the local frame scan position are corresponding to global scan position
- include:  timestamp, frame_index, scan_index, distance[m], x_global position, y_global_positon, x_local_position, y_local_positon
the file name is rosbagNumber_scenarioName.csv
- example format can be seen at /dataFiles/example_csv_format.csv and /dataFiles/example_yaml_format.yaml 


### YAML format
- rule: one yaml for one csv format. Must check the data in it.
- include: corresponding csv file name, laser min/max angle range, laser min/max distance range/whether laser coordinate is same with base_link frame, laser sensor frequency, opponent_odom publishing frequency
- If possible, check how long does it take to convert csv file 


### 2D scan heatmap conversion
Based on the image like at this site, https://www.nature.com/articles/s41467-019-12943-7/figures/4 we can visualize scan data and use that as a feature. Since it can be treated as an image, then we can use feature-descriptor method such as FAST, color histogram. So rather using csv file, we can make that as an image(just like occupangy grid map) per each frame. 
We chekced how long does it take to convert the given /scan data to the grid map. 

So we can say that if we have 360 degree distance data, then width will be 360(360 number of bins) and height will be distances. so \theta degree value is mapped to column vector, and the height of this column vector is maximum lidar scan distance(detection range). So when the lidar has a obstacle at 2m, then 0~2m value has increasing intensity, and no value is assigned above 2~maximum lidar scan distance. So black(0) at y = 0, and white(255) at y = detected distance. the value between is smootly increased(calculated automatically). Like an RGB channel, we can choose 3 channel values. (1) Distance, (2) Density(How many points are in that range), (3) Delta(current distance - previous distance). And then we can get a RGB color map. Since we have label(isWall, isOpponent, isFree, isStatic) each columns have a label. 
