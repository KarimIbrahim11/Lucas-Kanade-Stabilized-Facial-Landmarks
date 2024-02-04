# Lucas-Kanade-Stabilized-Facial-Landmarks
Implementing Lucas Kanade Optical Flow algorithm to stabilize the facial landmarks detected.

## Demo
Green landmarks is the stabilized LK stream while Red is the normal detections.

https://github.com/KarimIbrahim11/Lucas-Kanade-Stabilized-Facial-Landmarks/assets/47744559/37f6ca13-fabd-45c6-aa3f-5d3b559cdad7

## Results Duplication
To run the project please download and install conda from [ Anaconda Official Website ](https://www.anaconda.com/) then run the following commands:
```
git clone https://github.com/KarimIbrahim11/Lucas-Kanade-Stabilized-Facial-Landmarks.git
cd Lucas-Kanade-Stabilized-Facial-Landmarks
conda env create -f env.yaml
conda activate env
python main.py
```
Press [SPACE] to shuffle between the stable and unstable stream, [ESC] to exit the script.

> please note that this script is developed to work realtime using your desktop/laptop's camera.

## Experimental Results before Averaging


https://github.com/KarimIbrahim11/Lucas-Kanade-Stabilized-Facial-Landmarks/assets/47744559/acb6dc54-145b-4c49-b999-d673324ed878

