# Miliv2

## Description

Miliv2 is a Python-based project that implements a real-time object detection and tracking system, primarily focused on autonomous drone interception.

## Features

- Real-time object detection using YOLO.
- Drone tracking and interception.
- Sends relevant telemetry data to groundstation.

## Installation

git clone https://github.com/hasmorbyu/miliv2.git

cd miliv2

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

## Usage

python main.py

(in a new terminal for ground station telemetry data)
python station_listener.py 

## Exit Simulation

press Ctrl+C inside terminal to kill graphical rendering

## Visual Reference

<img width="1920" height="1006" alt="image" src="https://github.com/user-attachments/assets/2e07bf18-0bf2-406b-8fee-3e03fbce28be" />
