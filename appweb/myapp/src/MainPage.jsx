import React, { useContext } from 'react';
import { UserContext } from './UserContext'; // Import the context
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import UserInput from './UserInput';
import DailyGraphs from './DailyGraphs';
import Anomalies from './Anomalies';
import './App.css';

function MainPage() {
  const { userId } = useContext(UserContext); // Access userId from context

  return (
      <Router>
        <div className='App'>
          <header className='App-header'>
          <div > {userId ?
            <div className="user-box">
              <img src="user.png" alt="User Icon" style={{ width: '25px', height: '25px' }} />
              <p>{`${userId}`}</p>
            </div>
           :
            <div className="user-box">
              <img src="user.png" alt="User Icon" style={{ width: '25px', height: '25px' }} />
              <p><i>Undefined User</i></p>
            </div> } 
          </div>
            <div className='title'>
              {/* Maybe add an icon */}
              <h1>Smartwatch Anomaly Detection</h1>
            </div>
            <nav>
              <ul className='nav-tabs'>
                <li><NavLink to='/' className='tab-link'>
                  <div className='tab-names'>
                  <img src="userid.png" alt="Icon" style={{ width: '30px', height: '30px' }} />
                  <p>User ID</p>
                  </div>
                  </NavLink></li>
                <li><NavLink to='/dailygraphs' className='tab-link'>
                  <div className='tab-names'>
                    <img src="graphs.png" alt="Icon" style={{ width: '30px', height: '30px' }} />
                    <p>Daily Graphs</p>
                  </div>
                </NavLink></li>
                <li><NavLink to='/anomalies' className='tab-link'>
                  <div className='tab-names'>
                      <img src="anomalies.png" alt="Icon" style={{ width: '30px', height: '30px' }} />
                      <p>Anomalies</p>
                  </div>
                </NavLink></li>
              </ul>
            </nav>
          </header>
          <main className='background-image'>
            <Routes>
              <Route path='/' element={<UserInput />} />
              <Route path='/dailygraphs' element={<DailyGraphs />} />
              <Route path='/anomalies' element={<Anomalies />} />
            </Routes>
          </main>
        </div>
      </Router>
  );
}

export default MainPage;