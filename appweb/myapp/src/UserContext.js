import React, { createContext, useState } from 'react';

// Create a Context for the user ID
export const UserContext = createContext();

// Create a Provider component
export const UserProvider = ({ children }) => {
  const [userId, setUserId] = useState('');
  const [data, setData] = useState(null);
  const [days, setDays] = useState([]);
  const [anoms, setAnoms] = useState(null);

  return (
    <UserContext.Provider value={{ userId, setUserId, data, setData, days, setDays, anoms, setAnoms }}>
      {children}
    </UserContext.Provider>
  );
};