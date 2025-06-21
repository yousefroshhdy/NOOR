
import { Link } from 'react-router-dom';
import { Vehicle } from '@/services/mockData';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Battery, Gauge, MapPin } from 'lucide-react';
import { formatDistance } from '@/utils/formatters';

interface VehicleCardProps {
  vehicle: Vehicle;
}

const VehicleCard = ({ vehicle }: VehicleCardProps) => {
  // Status colors
  const statusColors = {
    driving: 'bg-green-500',
    parked: 'bg-blue-500',
    offline: 'bg-gray-500',
  };

  // Score colors
  const getScoreColor = (score?: number) => {
    if (!score) return 'text-gray-500';
    if (score >= 90) return 'text-green-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <Link to={`/vehicle/${vehicle.id}`}>
      <Card className="overflow-hidden h-full hover:shadow-md transition-shadow border border-gray-200">
        <div className="relative h-32 bg-gray-100">
          {vehicle.image ? (
            <img 
              src={vehicle.image} 
              alt={`${vehicle.make} ${vehicle.model}`} 
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-200">
              <span className="text-gray-400">No Image</span>
            </div>
          )}
          
          <div className="absolute bottom-2 left-2">
            <Badge 
              className={`${statusColors[vehicle.status]} text-white font-medium`}
            >
              {vehicle.status.charAt(0).toUpperCase() + vehicle.status.slice(1)}
            </Badge>
          </div>
        </div>
        
        <CardContent className="p-4">
          <h3 className="font-semibold text-lg mb-1">{vehicle.name}</h3>
          <p className="text-sm text-gray-500 mb-3">
            {vehicle.make} {vehicle.model} • {vehicle.year}
          </p>
          
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div className="flex flex-col items-center justify-center p-2 bg-gray-50 rounded">
              <Battery size={16} className="text-gray-500 mb-1" />
              <span className="font-medium">{vehicle.fuelLevel ? Math.round(vehicle.fuelLevel * 100) : 'N/A'}%</span>
              <span className="text-xs text-gray-500">Fuel</span>
            </div>
            
            <div className="flex flex-col items-center justify-center p-2 bg-gray-50 rounded">
              <Gauge size={16} className="text-gray-500 mb-1" />
              <span className={`font-medium ${getScoreColor(vehicle.drivingScore)}`}>
                {vehicle.drivingScore ?? 'N/A'}
              </span>
              <span className="text-xs text-gray-500">Score</span>
            </div>
            
            <div className="flex flex-col items-center justify-center p-2 bg-gray-50 rounded">
              <MapPin size={16} className="text-gray-500 mb-1" />
              <span className="font-medium">
                {vehicle.lastPosition ? 
                  formatDistance(vehicle.lastPosition.lat, vehicle.lastPosition.lng) : 
                  'Unknown'}
              </span>
              <span className="text-xs text-gray-500">Location</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
};

export default VehicleCard;
