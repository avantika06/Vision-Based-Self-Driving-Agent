# 🚗 Vision-Based Self-Driving Agent

**An autonomous driving system that combines Computer Vision (YOLOv8) with Rule-Based Decision Making to navigate complex traffic environments.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)

## 📌 Overview
This project simulates the core architecture of an autonomous vehicle. Unlike standard object detection demos that simply *label* objects, this system acts as an **Agent**: it perceives the environment, interprets spatial relationships, and executes driving commands (`ACCELERATE`, `BRAKE`, `STEER`) in real-time.

The system is built with **modular architecture**, separating the *Perception Layer* (Computer Vision) from the *Reasoning Layer* (Decision Logic), making it adaptable for future Reinforcement Learning (RL) integration.

## 🚀 Features
* **Real-Time Perception:** Uses **YOLOv8** to detect cars, pedestrians, traffic lights, and obstacles with high accuracy.
* **Spatial Awareness:** Implements a dynamic **"Danger Zone"** logic to differentiate between immediate threats and background noise.
* **Decision Making Agent:** A custom heuristic agent that reacts to:
    * **Traffic Lights:** Detects class `9` and initiates braking.
    * **Proximity:** Calculates distance to lead vehicles to prevent collisions.
    * **Lane Obstruction:** Decides to steer left/right based on obstacle position.
* **Interactive Dashboard:** A **Streamlit** UI that visualizes the "Brain" of the car, showing live telemetry, decision confidence, and the video feed.

## 🛠️ Tech Stack
* **Language:** Python
* **Vision Model:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Pre-trained on COCO)
* **Interface:** Streamlit
* **Image Processing:** OpenCV, NumPy

## 📂 Project Structure
```bash
├── agent.py            # The "Brain": Logic for spatial reasoning and decision making
├── app.py              # The "Body": Streamlit UI and YOLOv8 inference pipeline
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
\






## ⚙️ Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/yourusername/self-driving-agent.git](https://github.com/yourusername/self-driving-agent.git)
    cd self-driving-agent
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**

    ```bash
    streamlit run app.py
    ```

4.  **Usage:**

      * Once the app opens in your browser, upload any **MP4 dashcam video**.
      * (Recommended) Use a city driving video to test traffic light and pedestrian logic.

## 🧠 How It Works

The system operates on a **Perception-Action Loop**:

1.  **Input:** A frame from the dashcam video.
2.  **Perception (YOLOv8):** The model scans the frame and returns bounding boxes for relevant classes (Car, Person, Truck, Traffic Light).
3.  **State Estimation:**
      * The `agent.py` module calculates the center point of every object.
      * It determines if an object is inside the **"Danger Zone"** (the lane directly ahead).
4.  **Policy Execution:**
      * *If* Traffic Light detected $\rightarrow$ **BRAKE**
      * *If* Obstacle in Danger Zone AND Close $\rightarrow$ **BRAKE**
      * *If* Obstacle in Danger Zone AND Far $\rightarrow$ **STEER**
      * *Else* $\rightarrow$ **ACCELERATE**
5.  **Output:** The decision is overlaid on the video and displayed on the dashboard.

## 🔮 Currently working on

  * **Traffic Light Color Detection:** Use OpenCV HSV color masking to distinguish between Red (Stop) and Green (Go) lights.
  * **Reinforcement Learning:** Replace the heuristic `if/else` logic in `agent.py` with a trained **DQN or PPO model** that learns to drive via simulation.
  * **Lane Line Detection:** Integrate Canny Edge Detection to keep the car centered in the lane.

## 🤝 Contributing

Contributions are welcome\! Please feel free to submit a Pull Request.

```
```