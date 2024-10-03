import React, {} from 'react';
import { UserProvider } from './UserContext'; // Import the provider
import MainPage from './MainPage';
import './App.css';

function App() {
  return (
    <UserProvider>
      <MainPage />
    </UserProvider>
  );
}

export default App;