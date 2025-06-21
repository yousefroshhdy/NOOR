import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { isAuthenticated } from '@/services/authService';

const Index = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Redirect to dashboard if authenticated, otherwise to auth
    if (isAuthenticated()) {
      navigate('/dashboard');
    } else {
      navigate('/auth');
    }
  }, [navigate]);

  // This is just a placeholder while redirecting
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="text-center">
        <h1 className="text-2xl font-bold mb-4">DrivePulse Tracker</h1>
        <p className="text-gray-600">Loading application...</p>
      </div>
    </div>
  );
};

export default Index;
