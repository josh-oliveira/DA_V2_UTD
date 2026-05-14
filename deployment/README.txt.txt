#all project requirements must be installed on JETSON

must git clone Depth-anything-v2 links in prev readme. 

#DO NOT RUN a .pt    its  super slow on Jetson. 

python depth_cam.py --engine models/best_e400_L_1_03_01.pt --encoder vits

#USE
python depth_cam.py --engine models/best_e400_L_1_03_01.engine