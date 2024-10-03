import React, { useState, useContext } from 'react';
import { UserContext } from './UserContext'; // Import the context
import './Anomalies.css';

function Anomalies() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const { userId } = useContext(UserContext); // Access userId from context
  const { anoms, setAnoms } = useContext(UserContext); // Get anoms and setAnoms from context

  // Anomalies image path
  const imagePath = `http://localhost:3003/figures/user${userId}/_anom_days.png`;

  // Function to run the anomaly detection script
  const handleRunScript = async () => {
    setLoading(true);
    setError(null);
    setAnoms(null);
    try {
      const response = await fetch(`http://localhost:3003/anomdet?userId=${userId}`);
      if (!response.ok) {
        throw new Error('Error executing the script');
      }
      const result = await response.json();
      setAnoms(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Verify if there is a valid user to show the content */}
      {userId ? (
        <div>
            
          {/* RUN SCRIPT */}
          {anoms ? (
            <div className='initial-page'>
              <div className='slider-container'>

                <div className='img-container' style={{flexDirection: 'column'}}>

                  {/* Days Table */}
                  <table className='days-table'>
                    <tbody>
                      <tr>
                        <td className='first-column'>Anomalous Days</td>
                        {anoms.map((day, index) => (
                          <td key={index}>{day}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>

                  <img
                    src={imagePath}
                    className='image'
                  />

                </div>
              </div>
            </div>
          ) : (
            <div className='load-container'>

              <div className='message-container'>
                <p style={{marginBottom:0}}>In this tab you can take a look to the anomalous days detected by the system along with their heart rate graphs.</p>
                <p style={{marginTop:5}}>Please run the detection system first to analyze the latest collected data.</p>
              </div>

              <div className='button-box'>
                <button onClick={handleRunScript} className='load-button' disabled={loading}>
                  {loading ? 'Running...' : 'Run system'}
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

export default Anomalies;