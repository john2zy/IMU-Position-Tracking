# IMU Position Tracking
3D position tracking based on data from 9 degree of freedom IMU (Accelerometer, Gyroscope and Magnetometer). This can track orientation pretty accurately and position but with significant accumulated errors from double integration of acceleration.

## Project Structure
- `main.py`: where the main Extended Kalman Filter(EKF) and other algorithms sit.
- `butter.py`: a digital realtime butterworth filter implementation from [this repo](https://github.com/keikun555/Butter) with minor fixes. But I don't use realtime filtering now.
- `mathlib`: contains matrix definitions for the EKF and a filter helper function.
- `plotlib.py`: some wrappers for visualization used in prototyping.
- `main.ipynb`: almost the same as `main.py`, just used for prototyping.
- `/Ref`: References
- `/Doc`: an Algorithm description and a manual

# Data Source
I use an APP called [HyperIMU](https://play.google.com/store/apps/details?id=com.ianovir.hyper_imu) to pull (uncalibrated) data from my phone. Data is sent through TCP and received using `data_receiver.py`.
