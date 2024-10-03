import React, { useState, useContext } from 'react';
import { UserContext } from './UserContext'; // Import the context
import './UserInput.css';

function UserInput() {
  const [inputUserId, setInputUserId] = useState('');
  const [userMessage, setUserMessage] = useState('');

  const { userId, setUserId } = useContext(UserContext); // Get userId and setUserId from context
  const { setData } = useContext(UserContext); // Get setData from context
  const { setDays } = useContext(UserContext); // Get setDays from context
  const { setAnoms } = useContext(UserContext); // Get setAnoms from context

  // Users available in the project
  const validUsers = ['2022484408', '2347167796', '4020332650', '4388161847', '4558609924', '5553957443', '6117666160', '6962181067', '7007744171', '8792009665', '8877689391']
  
  // Function to handle the change in the input
  const handleInputChange = (e) => {
    setInputUserId(e.target.value);
  };

  // Function to handle when the confirm button is pressed
  const handleConfirm = () => {
    if (inputUserId.length !== 10 || isNaN(Number(inputUserId))) {
      setUserMessage('The user ID must be 10 digits')
      setUserId('');
      return;
    } if (!validUsers.includes(inputUserId)) {
      setUserMessage('The user ID is not valid')
      setUserId('');
    } else {
      setUserMessage('');
      setUserId(inputUserId);
      setData(null);
      setDays([]);
      setAnoms(null);
    }
  };

  return (
    <div className='user-input-page'>
        <div className='intro-container'>
          <p style={{marginBottom: 8}}>Welcome to the anomaly detection system for smartwatch data! This platform analyzes health data to 
            identify potential anomalies.</p>
          <p style={{marginTop: 0}}>Please enter the user ID of the patient you would like to review.</p>
        </div>
        <div className='user-input-container'>
            <h2>User ID</h2>
            <div className='input-wrapper'>
                <input
                type='text'
                className='user-input'
                placeholder='Enter user ID'
                value={inputUserId}
                onChange={handleInputChange}
                />
                <button onClick={handleConfirm} className='button'>Confirm</button>
            </div>
            {/* Show user message if it exists */}
            {userMessage && <p className='user-message'>{userMessage}</p>}

            {/* Show selected user ID if user ID exists */}
            {userId && (
            <div className='selected-user-container'>
              <p> Selected user: <strong>{userId}</strong> </p>
            </div>
            )}
        </div>
        
    </div>
  );
}

export default UserInput;