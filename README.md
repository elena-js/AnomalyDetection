# Anomaly Detection Using Machine Learning on Smartwatch Data

This project is designed to detect anomalies on smartwatch health data and deploy a web application to show the results.

## Table of Contents

- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Usage](#usage)

## Project Structure

```markdown
/anomalydetection
├── README.md
├── appweb/
│   ├── myapp/
│       ├── public/
│           └── ...
│       ├── server/
│           └── ...
│       ├── src/
│           └── ...
│       └── README.md
│       └── package.json
│       └── yarn.lock
├── dataset-fitness/
│   └── FitabaseData3.12.16-4.11.16-new.zip
│   └── FitabaseData4.12.16-5.12.16.zip
├── datasets/
│   └── data_user2022484408.pkl
│   └── ...
├── figures/
│   ├── user2022484408/
│       └── ...
│   ├── ...
│       └── ...
└── anom_det.py
└── preprocess.py
```

## File Descriptions

- **`README.md`**: This file provides an overview of the project, installation instructions, and usage guidelines.
- **`appweb/`**: This folder contains the project of the full-stack web app. Here is where `nmv` is installed.
  - **`myapp/`**: This is the folder of the project of the web app, which contains all its files. The frontend is built with React and the backend with Node.js.
    - **`public/`**: This folder contains static files such as `index.html`, `manifest.json`, `robots.txt`, images, and icons used in the application.
    - **`server/`**: This folder contains the backend (server-side) logic, including the entry point (`index.js`) and configuration files.
    - **`src/`**: This folder contains the frontend (client-side) logic, including all the React components and styles.
- **`dataset-fitness/`**: This folder contains the source datasets for the anomaly detection. There are 2 folders with all the files collected by Fitbit.
- **`datasets/`**: This folder contains the generated datasets after the preprocess phase (files with `.pkl` extension).
- **`figures/`**: This folder contains the generated figures from the preprocess and anomaly detection phases.
- **`anom_det.py`**: This file corresponds to the anomaly detection system.
- **`preprocess.py`**: This file corresponds to the preprocessing phase of the system.

*Note: The node_modules folder is not included.

## Installation

The steps required to install and set up the tools for the web application are described below:

### Frontend Setup:

- Install Node Version Manager (nvm) using the `nvm-setup.zip` file and then install Node.js version 18.7.0:
  
  ```bash
  nvm install 18.7.0
  ```
  
- Install Yarn version 1.22.17:
  
  ```bash
  yarn install
  ```
  
- Install the `create-react-app` module globally:

  ```bash
  yarn global add create-react-app
  ```
  
- Create the React application:

  ```bash
  yarn create react-app myapp
  ```
  
- To connect the frontend with the backend, add the proxy option in the `package.json` file:

  ```bash
  ”proxy”: ”http://localhost:3003”
  ```
  
- Navigate to the app folder and start the web application:

  ```bash
  cd myapp
  yarn start
  ```
  
  By default, the frontend runs on port 3000.

### Backend Setup:

- Inside the app folder (`myapp`), create the server directory:

  ```bash
  mkdir server
  ```
  
- Navigate to the `server` folder and initialize the project with default settings:

  ```bash
  cd server
  yarn init -y
  ```
  
- Install Express and subsequently any additional dependencies required for the backend:

  ```bash
  yarn add express
  ```
  
- Create the `index.js` file with the backend code (specify the server listening port as 3003).
  
- Start running the server:

  ```bash
  yarn start
  ```

## Usage

To run the project using the web app, simply execute the following commands:

- In `myapp` folder:
  
  ```bash
  yarn start
  ```

- In `server` folder:

  ```bash
  yarn start
  ```

  On the other hand, if you want to execute only the system using the python files, execute in the main folder:

- For the `preprocess` phase:
  
  ```bash
  python preprocess.py [user] [show_output]
  ```

Change `[user]` for the user id (e.g. `2022484408`).

Change `[show_output]` for `true` (if you want to see the system comments along the execution) or `false` (if you don't). You can also skip this parameter and the system won't show the comments.

- In a similar way, for the `anomaly detection` phase:
  
  ```bash
  python anom_det.py [user] [show_output]
  ```
