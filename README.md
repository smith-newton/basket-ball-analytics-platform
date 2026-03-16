
# Basketball Analytics Platform

A computer vision-based basketball analytics project built with Python and YOLO for detecting and analyzing basketball gameplay from video input.

## Features

- Player and object detection using YOLO
- Video processing pipeline for basketball match analysis
- Model inference with pretrained weights
- Output video generation with processed analytics
- Simple Python-based project structure

## Project Structure

```bash
basket-ball-analytics-platform/
│── src/
│── app.py
│── yolov11.py
│── best11.pt
│── processed_output.mp4
│── requirements.txt
````

## Tech Stack

* Python
* YOLO
* OpenCV
* Deep Learning / Computer Vision

## How It Works

1. Input basketball video is provided to the system.
2. The YOLO model detects players / ball / important objects in each frame.
3. Frames are processed for analytics and tracking.
4. The final processed video is generated as output.

## Installation

Clone the repository:

```bash
git clone https://github.com/smith-newton/basket-ball-analytics-platform.git
cd basket-ball-analytics-platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main application:

```bash
python app.py
```

If needed, you can also run:

```bash
python yolov11.py
```

## Output

* Processed basketball analytics video will be generated
* Example output file:

  * `processed_output.mp4`

## Requirements

Make sure you have:

* Python 3.9+
* Required Python libraries from `requirements.txt`

## Future Improvements

* Player tracking with IDs
* Team classification
* Shot detection
* Pass analysis
* Heatmaps and movement analytics
* Real-time dashboard support

## Repository Info

This repository currently includes:

* `src` folder
* `app.py`
* `yolov11.py`
* `best11.pt`
* `processed_output.mp4`
* `requirements.txt` ([GitHub][1])

## Author

**Smith Newton K**

## Files
- `app.py` – main application
- `yolov11.py` – YOLO-based detection module
- `best11.pt` – trained model weights
- `processed_output.mp4` – sample processed output
- `requirements.txt` – dependencies

## Installation
```bash
git clone https://github.com/smith-newton/basket-ball-analytics-platform.git
cd basket-ball-analytics-platform
pip install -r requirements.txt
````

## Run

```bash
python app.py
```

## Tech Stack

* Python
* YOLO
* OpenCV
* Computer Vision

## Future Scope

* Player tracking
* Team classification
* Shot and pass analytics
* Real-time insights


