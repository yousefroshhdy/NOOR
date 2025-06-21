
import { formatDistanceToNow } from 'date-fns';

// Format distance in miles/km from coordinates (dummy implementation)
export const formatDistance = (lat: number, lng: number): string => {
  // In a real app, this would calculate actual distance from user's location
  // For now, just return a random distance for demo purposes
  const randomDistance = Math.floor(Math.random() * 10) + 1;
  return `${randomDistance} mi`;
};

// Format timestamp as relative time (e.g., "2 hours ago")
export const formatRelativeTime = (timestamp: number): string => {
  return formatDistanceToNow(timestamp, { addSuffix: true });
};

// Format date as MM/DD/YYYY
export const formatDate = (timestamp: number): string => {
  const date = new Date(timestamp);
  return `${date.getMonth() + 1}/${date.getDate()}/${date.getFullYear()}`;
};

// Format time as HH:MM AM/PM
export const formatTime = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
};

// Format speed in mph
export const formatSpeed = (speedMph?: number): string => {
  if (speedMph === undefined || speedMph === null) return 'N/A';
  return `${Math.round(speedMph)} mph`;
};

// Format battery/fuel level as percentage
export const formatBatteryLevel = (level?: number): string => {
  if (level === undefined || level === null) return 'N/A';
  return `${Math.round(level * 100)}%`;
};

// Format driving score with color indicator
export const formatDrivingScore = (score?: number): { text: string; color: string } => {
  if (score === undefined || score === null) {
    return { text: 'N/A', color: 'text-gray-500' };
  }
  
  let color = 'text-gray-500';
  if (score >= 90) color = 'text-green-600';
  else if (score >= 70) color = 'text-yellow-600';
  else color = 'text-red-600';
  
  return { text: score.toString(), color };
};
