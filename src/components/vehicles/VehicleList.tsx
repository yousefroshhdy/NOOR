
import { useState, useEffect } from 'react';
import { Vehicle } from '@/services/mockData';
import { getVehicles, subscribeToVehicleUpdates } from '@/services/vehicleService';
import VehicleCard from './VehicleCard';
import { Input } from '@/components/ui/input';
import { Search } from 'lucide-react';

const VehicleList = () => {
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const fetchVehicles = async () => {
      setIsLoading(true);
      try {
        const data = await getVehicles();
        setVehicles(data);
      } catch (error) {
        console.error('Failed to fetch vehicles:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchVehicles();

    // Subscribe to real-time updates
    const unsubscribe = subscribeToVehicleUpdates((updatedVehicle) => {
      setVehicles((prevVehicles) =>
        prevVehicles.map((vehicle) =>
          vehicle.id === updatedVehicle.id ? updatedVehicle : vehicle
        )
      );
    });

    return () => {
      unsubscribe();
    };
  }, []);

  // Filter vehicles based on search term
  const filteredVehicles = vehicles.filter((vehicle) => {
    const searchString = searchTerm.toLowerCase();
    return (
      vehicle.name.toLowerCase().includes(searchString) ||
      vehicle.make.toLowerCase().includes(searchString) ||
      vehicle.model.toLowerCase().includes(searchString) ||
      vehicle.licensePlate.toLowerCase().includes(searchString)
    );
  });

  // Sort vehicles: driving first, then parked, then offline
  const sortedVehicles = [...filteredVehicles].sort((a, b) => {
    const statusOrder = { driving: 0, parked: 1, offline: 2 };
    return statusOrder[a.status] - statusOrder[b.status];
  });

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
        <Input
          type="text"
          placeholder="Search vehicles..."
          className="pl-10"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-64 rounded-lg bg-gray-100 animate-pulse"
            />
          ))}
        </div>
      ) : sortedVehicles.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pb-20">
          {sortedVehicles.map((vehicle) => (
            <VehicleCard key={vehicle.id} vehicle={vehicle} />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-gray-500">No vehicles found</p>
        </div>
      )}
    </div>
  );
};

export default VehicleList;
