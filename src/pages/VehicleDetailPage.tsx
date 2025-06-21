
import { useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { isAuthenticated } from '@/services/authService';
import AppLayout from '@/components/layout/AppLayout';
import VehicleDetail from '@/components/vehicles/VehicleDetail';

const VehicleDetailPage = () => {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();

  // Check if user is authenticated
  useEffect(() => {
    if (!isAuthenticated()) {
      navigate('/auth');
    }
  }, [navigate]);

  return (
    <AppLayout title="Vehicle Details">
      <div className="max-w-4xl mx-auto">
        <VehicleDetail />
      </div>
    </AppLayout>
  );
};

export default VehicleDetailPage;
