# Anomaly Detection Using Machine Learning on Smartwatch Data

This project is designed to detect anomalies on smartwatch health data and deploy a web application to show the results.

## Table of Contents

- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Usage](#usage)

## Project Structure

/my-project
├── README.md
├── appweb/
│   ├── myapp/
│       └── public/
│       └── server/
│       └── src/
├── dataset-fitness/
│   ├── FitabaseData3.12.16-4.11.16-new/
│       └── ...
│   ├── FitabaseData4.12.16-5.12.16/
│       └── ...  
├── datasets/
│   └── data_user2022484408.pkl
│   └── ...
├── figures/
│   ├── user2022484408/
│       └── ...
└── anom_det.py
└── preprocess.py

## File Descriptions

- **`README.md`**: This file provides an overview of the project, installation instructions, and usage guidelines.
- **`appweb/`**: Folder that contains the project of the web app.
  - **`myapp/`**: Folder of the project of the web app. It contains all the files of the app web.
    - **`public/`**: Contains static files such as `index.html`, `manifest.json`, `robots.txt`, images, and icons used in the application.
    - **`server/`**: Contains server-side logic, including the entry point (`index.js`) and configuration files for the Node.js application.
    - **`src/`**:
- **`dataset-fitness/`**: Folder with the source datasets for the anomaly detection. It contains 2 folders with all the files collected by Fitbit.
- **`datasets/`**: Folder with the generated datasets after the preprocess (files with .pkl extension).
- **`figures/`**: Folder with the generated figures from the preprocess and anomaly detection phases.
- **`anom_det.py`**: The python file with the anomaly detection system.
- **`preprocess.py`**: The python file with the preprocessing phase.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
To run the project, execute the main script as follows:
