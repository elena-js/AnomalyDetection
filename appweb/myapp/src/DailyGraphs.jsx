import React, { useState, useEffect, useContext } from 'react';
import { UserContext } from './UserContext'; // Import the context
import './DailyGraphs.css';

function DailyGraphs() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTab, setSelectedTab] = useState('HR');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [imagePaths, setImagePaths] = useState([]);
  const [oneGraphicImg, setOneGraphicImg] = useState(false); // Start seing images with time values

  const { userId } = useContext(UserContext); // Access userId from context
  const { data, setData } = useContext(UserContext); // Get data and setData from context
  const { days, setDays } = useContext(UserContext); // Get days and setDays from context

  // Base path for the images
  const basePath = `http://localhost:3003/figures/user${userId}/`;

  // Image paths (HR representation)
  const imagePathsHR = days.map(day => `${basePath}${day}.png`);
  // Image paths (HR/INT representation)
  const imagePathsHRINT = days.map(day => `${basePath}hr_int_${day}.png`);
  const imagePathsHRINTtime = days.map(day => `${basePath}hr_int_time_${day}.png`);
  // Image paths (HR/STEP representation)
  const imagePathsHRSTEP = days.map(day => `${basePath}hr_step_${day}.png`);
  const imagePathsHRSTEPtime = days.map(day => `${basePath}hr_step_time_${day}.png`);

  // Function to handle tab clicks
  const handleTabClick = (tab) => {
    setSelectedTab(tab);
  };

  // Function to set timeImg true if the checkbox is checked and viceversa
  const handleChange = (event) => {
    setOneGraphicImg(event.target.checked);
  };

  // Function to go to the next image
  const nextImage = () => {
    if (currentIndex < imagePaths.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  // Function to go to the previous image
  const prevImage = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  // Function to run the preprocess script
  const handleRunScript = async () => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const response = await fetch(`http://localhost:3003/preprocess?userId=${userId}`);
      if (!response.ok) {
        throw new Error('Error executing the script');
      }
      const result = await response.json();
      setData(result);
      const extractedDays = result.map(item => item.Date); // Extract Date column (YYYY-MM-DD)
      setDays(extractedDays);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Update imagePaths based on selectedTab
  useEffect(() => {
    switch (selectedTab) {
      case 'HR':
        setImagePaths(imagePathsHR);
        break;
      case 'INT':
        // If the box is checked, show one graphic image, 
        // if not show it with time values
        if (oneGraphicImg) {
          setImagePaths(imagePathsHRINT);
        } else {
          setImagePaths(imagePathsHRINTtime);
        }
        break;
      case 'STEP':
        if (oneGraphicImg) {
          setImagePaths(imagePathsHRSTEP);
        } else {
          setImagePaths(imagePathsHRSTEPtime);
        }
        break;
      default:
        setImagePaths([imagePathsHR]);
        break;
    }
  }, [selectedTab, oneGraphicImg, data]);

  return (
    <div>
      {/* Verify if there is a valid user to show the content */}
      {userId ? (
        <div>

          {/* RUN SCRIPT */}
          {data ? (
            <div className='initial-page'>
              <div className='slider-container'>

                <div className='tabs-bar'>
                  <nav className='tabs-nav'>
                    <button className={`tab-button ${selectedTab === 'HR' ? 'active' : 'img-tab-button'}`}
                      onClick={() => handleTabClick('HR')} > HeartRate </button>
                    <button
                      className={`tab-button ${selectedTab === 'INT' ? 'active' : 'img-tab-button'}`}
                      onClick={() => handleTabClick('INT')} > Intensities </button>
                    <button
                      className={`tab-button ${selectedTab === 'STEP' ? 'active' : 'img-tab-button'}`}
                      onClick={() => handleTabClick('STEP')} > Steps </button>
                  </nav>
                </div>

                {selectedTab!='HR' ? (
                <label className="custom-checkbox">
                  <input type="checkbox" checked={oneGraphicImg} onChange={handleChange} />
                  <span className="checkbox"></span> Show feature relations in one graphic
                </label>
                ) : (
                  <label className="custom-checkbox">
                  <p style={{marginBottom: 0, marginTop: 8.5, fontSize: 12}}>Note: The red lines represent the thresholds of normal values at rest.</p>
                  </label>
                )
                }

                <div className='img-container'>
                  <button onClick={prevImage} disabled={currentIndex === 0} className={`img-button ${currentIndex === 0 ? 
                    'button-disabled' : ''}`}>&lt;</button>
                  <img
                    src={imagePaths[currentIndex]}
                    alt={`Slide ${currentIndex + 1}`}
                    className='image'
                  />
                  <button onClick={nextImage} disabled={currentIndex === imagePaths.length - 1} className={`img-button 
                    ${currentIndex === imagePaths.length - 1 ? 'button-disabled' : ''}`}>&gt;</button>
                </div>
              </div> 
            </div>
          ) : (
            <div className='load-container'>
              <div className='message-container'>
                <p style={{marginBottom:0}}>Here you can view the graphical representation of the patient's health data.</p>
                <p style={{marginTop:5}}>Before displaying the following images, we need to load the latest data collected.</p>
              </div>
              <div className='button-box'>
                <button onClick={handleRunScript} className='load-button' disabled={loading}>
                  {loading ? 'Loading...' : 'Load Data'}
                </button>
              {loading && (
                  <div className='spinner-container'>
                      <p className='loading-message'>This may take a few minutes...</p>
                      <div className='spinner'></div>
                  </div>
              )}
              </div>
            </div>
          )}
          {error && <p>Error: {error}</p>}
      </div>
      ) : (
        <div className='not-user-container'>
          <p>USER NOT SELECTED</p>
        </div>
      )}
    </div>
  );
}

export default DailyGraphs;