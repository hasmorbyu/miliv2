# Miliv2

## Description

Miliv2 is a Python-based project that implements a real-time object detection and tracking system, primarily focused on autonomous drone interception.

## Features

- Real-time object detection using YOLO.
- Drone tracking and interception.
- Sends relevant telemetry data to groundstation.

## Installation

git clone https://github.com/yourusername/miliv2.git
cd miliv2

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

## Usage

python main.py

(in a new terminal for ground station telemetry data)
python station_listener.py 

