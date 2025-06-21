
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { isAuthenticated } from '@/services/authService';
import AppLayout from '@/components/layout/AppLayout';
import VehicleList from '@/components/vehicles/VehicleList';

const Dashboard = () => {
  const navigate = useNavigate();

  // Check if user is authenticated
  useEffect(() => {
    if (!isAuthenticated()) {
      navigate('/auth');
    }
  }, [navigate]);

  return (
    <AppLayout title="My Vehicles">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">My Vehicles</h1>
        <VehicleList />
      </div>
    </AppLayout>
  );
};

export default Dashboard;
